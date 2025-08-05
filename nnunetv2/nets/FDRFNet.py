from typing import Union, Type, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from .utility.helper import convert_conv_op_to_dim


from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from .utility.helper import convert_conv_op_to_dim
from .utility.plain_conv_encoder import PlainConvEncoder
from .utility.residual import BasicBlockD, BottleneckD
from .utility.residual_encoders import ResidualEncoder, SpecificResidualEncoder
from .utility.unet_decoder import UNetDecoder
from .utility.unet_residual_decoder import UNetResDecoder, SpecificUNetResDecoder
from .utility.helper import get_matching_instancenorm, convert_dim_to_conv_op
from .utility.weight_init import init_last_bn_before_add_to_0
from .utility.network_initialization import InitWeights_He
from .utility.weight_init import init_last_bn_before_add_to_0

from copy import deepcopy


from torch.cuda.amp import autocast
from .utility.residual import BasicBlockD

def concatenate_multimodal_features(
    features_by_modality: List[List[torch.Tensor]]
) -> List[torch.Tensor]:
    """
    将来自不同模态、不同层级的特征在通道维度上进行拼接。

    假设每个模态在每个层级都提取了特征，且所有模态在同一层级具有
    相同的空间维度 (D, H, W)。

    Args:
        features_by_modality (List[List[torch.Tensor]]):
            一个包含所有模态特征的列表。
            外部列表的每个元素代表一个模态。
            内部列表的每个元素代表该模态在不同层级（从浅到深）提取到的特征。
            例如：[[mod1_feat_L1, mod1_feat_L2], [mod2_feat_L1, mod2_feat_L2]]

    Returns:
        List[torch.Tensor]:
            一个列表，包含每个层级融合（拼接）后的特征。
            列表中每个元素的通道数将是对应层级所有模态通道数之和。
    """
    if not features_by_modality:
        return []

    num_modalities = len(features_by_modality)
    num_levels = len(features_by_modality[0]) # 假设所有模态的层级数量相同

    fused_features_per_level: List[torch.Tensor] = []

    # 遍历每个层级 (例如，从最浅层到最深层)
    for level_idx in range(num_levels):
        # 收集当前层级所有模态的特征
        features_at_current_level = []
        for mod_idx in range(num_modalities):
            # 确保当前模态在当前层级有特征
            if level_idx < len(features_by_modality[mod_idx]):
                features_at_current_level.append(features_by_modality[mod_idx][level_idx])
            else:
                raise ValueError(
                    f"模态 {mod_idx} 在层级 {level_idx} 缺少特征。"
                    "请确保所有模态在所有层级都有对应的特征。"
                )

        # 在通道维度 (dim=1) 上拼接当前层级的所有模态特征
        # 例如，如果每个模态在该层级输出 C 通道，M 个模态拼接后就是 M*C 通道
        concatenated_feature = torch.cat(features_at_current_level, dim=1)
        fused_features_per_level.append(concatenated_feature)

    return fused_features_per_level

class HighFrequencyFeatureExtractor(nn.Module):
    def __init__(self, cutoff_ratio: float = 0.05):
        """
        初始化高频特征提取器。
        
        Args:
            cutoff_ratio (float): 用于高斯高通滤波器的截止频率比率。
                                  该值越小，保留的低频信息越多，高频信息越少。
                                  建议范围 0.01 到 0.1。
        """
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

    def _create_3d_gaussian_high_pass_filter(self, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        创建 3D 高斯高通滤波器。
        
        Args:
            D (int): 深度维度大小。
            H (int): 高度维度大小。
            W (int): 宽度维度大小。
            device (torch.device): 张量所在的设备 (CPU 或 CUDA)。
            
        Returns:
            torch.Tensor: 形状为 (D, H, W) 的 3D 高斯高通滤波器。
        """
        # 创建频率网格，范围通常为 [-0.5, 0.5)
        d_coords = torch.fft.fftfreq(D, device=device)
        h_coords = torch.fft.fftfreq(H, device=device)
        w_coords = torch.fft.fftfreq(W, device=device)
        
        # 使用 torch.meshgrid 创建 3D 频率坐标网格
        # indexing='ij' 确保维度顺序与 (D, H, W) 匹配
        d_grid, h_grid, w_grid = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        
        # 计算每个点到频率中心 (0,0,0) 的归一化距离的平方
        distance_sq = d_grid**2 + h_grid**2 + w_grid**2
        
        # 定义高斯高通滤波器：G(d) = 1 - exp(-d^2 / (2*sigma^2))
        # sigma 决定了滤波器过渡区域的平滑度，通常与截止频率相关
        sigma = self.cutoff_ratio / 3.0 # 经验值，可以根据实验调整
        
        gaussian_low_pass = torch.exp(-distance_sq / (2 * sigma**2))
        high_pass_filter = 1 - gaussian_low_pass
        
        # 确保滤波器值在 [0, 1] 之间
        high_pass_filter = torch.clamp(high_pass_filter, 0.0, 1.0)
        
        return high_pass_filter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，提取高频特征并与原始输入拼接。
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 (b, c, h, w, d)。
                              数据类型应为 torch.float32。
                               
        Returns:
            torch.Tensor: 包含原始模态和高频模态的拼接特征，形状为 (b, 2*c, h, w, d)。
        """
        if x.dim() != 5:
            raise ValueError(f"输入张量维度必须是 5 (b, c, h, w, d)，但得到 {x.dim()} 维。")
        
        # 提取输入张量的维度信息
        b, c, h, w, d = x.shape
        device = x.device # 获取张量所在的设备
        
        # 1. 对输入张量进行 3D 傅里叶变换
        # dim=(-3, -2, -1) 表示在 h, w, d 维度上进行 FFT
        # 输出的频域数据是复数类型 (complex64)
        freq_data = torch.fft.fftn(x, dim=(-3, -2, -1))
        
        # 2. 将零频率成分移到中心，便于滤波器设计和应用
        freq_data_shifted = torch.fft.fftshift(freq_data, dim=(-3, -2, -1))
        
        # 3. 创建并应用 3D 高斯高通滤波器
        # 滤波器仅在 h, w, d 维度上创建，通过广播机制应用于批次和通道
        high_pass_filter = self._create_3d_gaussian_high_pass_filter(h, w, d, device)
        
        # 乘以滤波器：PyTorch 会自动将 (D, H, W) 形状的滤波器广播到 (B, C, D, H, W) 形状的频域数据
        filtered_freq_data_shifted = freq_data_shifted * high_pass_filter
        
        # 4. 将零频率成分移回其原始位置，为逆傅里叶变换做准备
        filtered_freq_data = torch.fft.ifftshift(filtered_freq_data_shifted, dim=(-3, -2, -1))
        
        # 5. 进行 3D 逆傅里叶变换，将高频信息转回空间域
        # 输出仍为复数类型 (complex64)
        high_freq_spatial = torch.fft.ifftn(filtered_freq_data, dim=(-3, -2, -1))
        
        # 傅里叶逆变换的实部即为我们所需的高频特征
        # 转换为 torch.float32 类型，确保与原始输入数据类型一致
        high_freq_features = high_freq_spatial.real.to(torch.float32)
        
        # 6. 将原始输入特征与提取出的高频特征沿通道维度拼接
        # 原始形状 (b, c, h, w, d)
        # 高频特征形状 (b, c, h, w, d)
        # 拼接后形状 (b, 2*c, h, w, d)
        output_features = torch.cat((x, high_freq_features), dim=1)
        
        return output_features
    
class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size = 3, stride = 1, padding = 1, pad_type = 'zeros', norm = 'in', is_training = True, act_type = 'lrelu', relufactor = 0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels = in_ch, out_channels = out_ch, kernel_size = k_size, stride = stride, padding = padding, padding_mode = pad_type, bias = True)
        
        self.dropout = nn.Dropout3d(p = 0.1, inplace=True)
        self.norm = nn.BatchNorm3d(in_ch)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope = relufactor, inplace = True)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.dropout(x)
        # x = self.conv(x)
        # x = self.dropout(x)
        # x = self.norm(x)
        # x = self.activation(x)
        return x
    
class SingalModalEncoder(nn.Module):
    # def __init__(self, input_size, input_channels, n_stages, basic_dims = 4):
    def __init__(self, basic_dims = 4):
        # super(MultiModalEncoder, self).__init__()
        super().__init__()
        self.e1_c1 = nn.Conv3d(in_channels = 1, out_channels = basic_dims * 2, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'reflect', bias = True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type = 'reflect')
        self.e1_c3 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type = 'reflect')

        self.e2_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride = 2, pad_type = 'reflect')
        self.e2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 20, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(basic_dims * 20, basic_dims * 20, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(basic_dims * 20, basic_dims * 20, pad_type='reflect')

        self.e6_c1 = general_conv3d_prenorm(basic_dims * 20, basic_dims * 20, stride=2, pad_type='reflect')
        self.e6_c2 = general_conv3d_prenorm(basic_dims * 20, basic_dims * 20, pad_type='reflect')
        self.e6_c3 = general_conv3d_prenorm(basic_dims * 20, basic_dims * 20, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        x6 = self.e6_c1(x5)
        x6 = x6 + self.e6_c3(self.e6_c2(x6))

        return x1, x2, x3, x4, x5, x6

class MultiModalEncoder_Brats(nn.Module):
    def __init__(self, modal_num = 4):
        super().__init__()
        self.modal_num = modal_num
        self.t1_encoder = SingalModalEncoder()
        self.t1ce_encoder = SingalModalEncoder()
        self.t2_encoder = SingalModalEncoder()
        self.flair_encoder = SingalModalEncoder()

    def forward(self, x):
        t1_features = self.t1_encoder(x[:, 0, ...].unsqueeze(1))
        t1ce_features = self.t1ce_encoder(x[:, 1, ...].unsqueeze(1))
        t2_features = self.t2_encoder(x[:, 2, ...].unsqueeze(1))
        flair_features = self.flair_encoder(x[:, 3, ...].unsqueeze(1))

        return t1_features, t1ce_features, t2_features, flair_features    

class MultiModalEncoder_MMWHS(nn.Module):
    def __init__(self, modal_num = 2):
        super().__init__()
        self.modal_num = modal_num
        self.ct_encoder = SingalModalEncoder(basic_dims = 8)
        self.mri_encoder = SingalModalEncoder(basic_dims = 8)
    def forward(self, x):
        ct_features = self.ct_encoder(x[:, 0, ...].unsqueeze(1))
        mri_features = self.mri_encoder(x[:, 1, ...].unsqueeze(1))

        return ct_features, mri_features
    
class MultiModalEncoder(nn.Module):
    def __init__(self, modal_num = 4):
        super().__init__()
        self.modal_num = modal_num

        self.e1_c1 = nn.Conv3d(in_channels = modal_num, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'reflect', bias = True)
        self.e1_c2 = general_conv3d_prenorm(32, 32, pad_type = 'reflect')
        self.e1_c3 = general_conv3d_prenorm(32, 32, pad_type = 'reflect')

        self.e2_c1 = general_conv3d_prenorm(32, 64, stride = 2, pad_type = 'reflect')
        self.e2_c2 = general_conv3d_prenorm(64, 64, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(64, 64, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(64, 128, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(128, 128, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(128, 128, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm(128, 256, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(256, 256, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(256, 256, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm(256, 320, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(320, 320, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(320, 320, pad_type='reflect')

        self.e6_c1 = general_conv3d_prenorm(320, 320, stride=2, pad_type='reflect')
        self.e6_c2 = general_conv3d_prenorm(320, 320, pad_type='reflect')
        self.e6_c3 = general_conv3d_prenorm(320, 320, pad_type='reflect')
        
    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        x6 = self.e6_c1(x5)
        x6 = x6 + self.e6_c3(self.e6_c2(x6))

        return x1, x2, x3, x4, x5, x6
    
class SharedDecoder(nn.Module):
    def __init__(self, basic_dims = 4):
        super().__init__()
        self.basic_dims = basic_dims
        self.c = self.basic_dims // 4
        self.d5 = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        self.d5_c1 = general_conv3d_prenorm(basic_dims * 20, basic_dims * 20, pad_type = 'reflect')
        self.d5_c2 = general_conv3d_prenorm(basic_dims * 40, basic_dims * 20, pad_type = 'reflect')
        self.d5_out = general_conv3d_prenorm(basic_dims * 20, basic_dims * 20, k_size = 1, padding = 0, pad_type = 'reflect')

        self.d4 = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims * 20, basic_dims * 20, pad_type = 'reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 36, basic_dims * 16, pad_type = 'reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, k_size = 1, padding = 0, pad_type = 'reflect')

        self.d3 = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type = 'reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 24, basic_dims * 8, pad_type = 'reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size = 1, padding = 0, pad_type = 'reflect')

        self.d2 = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type = 'reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 12, basic_dims * 4, pad_type = 'reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size = 1, padding = 0, pad_type = 'reflect')

        self.d1 = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type = 'reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 6, basic_dims * 2, pad_type = 'reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size = 1, padding = 0, pad_type = 'reflect')

        self.conv3d = nn.Conv3d(in_channels = basic_dims * 2, out_channels = 1, kernel_size = 1, stride = 1, padding = 0, bias = True)

        # self.fusion1 = general_conv3d_prenorm(basic_dims * 4 // self.c + 2 * basic_dims * 4 // self.c, basic_dims * 4 // self.c, pad_type = 'reflect')
        # self.fusion2 = general_conv3d_prenorm(basic_dims * 8 // self.c  + 2 * basic_dims * 8 // self.c, basic_dims * 8 // self.c, pad_type = 'reflect')
        # self.fusion3 = general_conv3d_prenorm(basic_dims * 16 // self.c + 2 * basic_dims * 16 // self.c, basic_dims * 16 // self.c, pad_type = 'reflect')
        # self.fusion4 = general_conv3d_prenorm(basic_dims * 32 // self.c + 2 * basic_dims * 32 // self.c, basic_dims * 32 // self.c, pad_type = 'reflect')
        # self.fusion5 = general_conv3d_prenorm(basic_dims * 40 // self.c + 2 * basic_dims * 40 // self.c, basic_dims * 40 // self.c, pad_type = 'reflect')
        # self.fusion6 = general_conv3d_prenorm(basic_dims * 40 // self.c + 2 * basic_dims * 40 // self.c, basic_dims * 40 // self.c, pad_type = 'reflect')
        if basic_dims == 8:
            self.fusion1 = nn.Conv3d(basic_dims * 4 // self.c + 2 * basic_dims * 4 // self.c, basic_dims * 4 // self.c, kernel_size = 1, stride = 1, padding = 0, bias = True)
            self.fusion2 = nn.Conv3d(basic_dims * 8 // self.c + 2 * basic_dims * 8 // self.c, basic_dims * 8 // self.c, kernel_size = 1, stride = 1, padding = 0, bias = True)
            self.fusion3 = nn.Conv3d(basic_dims * 16 // self.c + 2 * basic_dims * 16 // self.c, basic_dims * 16 // self.c, kernel_size = 1, stride = 1, padding = 0, bias = True)
            self.fusion4 = nn.Conv3d(basic_dims * 32 // self.c + 2 * basic_dims * 32 // self.c, basic_dims * 32 // self.c, kernel_size = 1, stride = 1, padding = 0, bias = True)
            self.fusion5 = nn.Conv3d(basic_dims * 40 // self.c + 2 * basic_dims * 40 // self.c, basic_dims * 40 // self.c, kernel_size = 1, stride = 1, padding = 0, bias = True)
            self.fusion6 = nn.Conv3d(basic_dims * 40 // self.c + 2 * basic_dims * 40 // self.c, basic_dims * 40 // self.c, kernel_size = 1, stride = 1, padding = 0, bias = True)
        elif basic_dims == 4:
            self.fusion1 = nn.Conv3d(basic_dims * 4 // (2 * self.c) + 2 * basic_dims * 4 // self.c, basic_dims * 4 // (2 * self.c), kernel_size = 1, stride = 1, padding = 0, bias = True)
            self.fusion2 = nn.Conv3d(basic_dims * 8 // (2 * self.c) + 2 * basic_dims * 8 // self.c, basic_dims * 8 // (2 * self.c), kernel_size = 1, stride = 1, padding = 0, bias = True)
            self.fusion3 = nn.Conv3d(basic_dims * 16 // (2 * self.c) + 2 * basic_dims * 16 // self.c, basic_dims * 16 // (2 * self.c), kernel_size = 1, stride = 1, padding = 0, bias = True)
            self.fusion4 = nn.Conv3d(basic_dims * 32 // (2 * self.c) + 2 * basic_dims * 32 // self.c, basic_dims * 32 // (2 * self.c), kernel_size = 1, stride = 1, padding = 0, bias = True)
            self.fusion5 = nn.Conv3d(basic_dims * 40 // (2 * self.c) + 2 * basic_dims * 40 // self.c, basic_dims * 40 // (2 * self.c), kernel_size = 1, stride = 1, padding = 0, bias = True)
            self.fusion6 = nn.Conv3d(basic_dims * 40 // (2 * self.c) + 2 * basic_dims * 40 // self.c, basic_dims * 40 // (2 * self.c), kernel_size = 1, stride = 1, padding = 0, bias = True)

    def forward(self, shared_f, x1, x2, x3, x4, x5, x6):
        x6 = self.fusion6(torch.cat([shared_f[5], x6], dim = 1))
        de_x6 = self.d5_c1(self.d5(x6))

        x5 = self.fusion5(torch.cat([shared_f[4], x5], dim = 1))
        cat_x5 = torch.cat((de_x6, x5), dim = 1)
        de_x5 = self.d5_out(self.d5_c2(cat_x5))
        de_x5 = self.d4_c1(self.d4(de_x5))

        x4 = self.fusion4(torch.cat([shared_f[3], x4], dim = 1))
        cat_x4 = torch.cat((de_x5, x4), dim = 1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        x3 = self.fusion3(torch.cat([shared_f[2], x3], dim = 1))
        cat_x3 = torch.cat((de_x4, x3), dim = 1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        x2 = self.fusion2(torch.cat([shared_f[1], x2], dim = 1))
        cat_x2 = torch.cat((de_x3, x2), dim = 1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        x1 = self.fusion1(torch.cat([shared_f[0], x1], dim = 1))
        cat_x1 = torch.cat((de_x2, x1), dim = 1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        predicted_img = self.conv3d(de_x1)

        return predicted_img
    
class FusionUNetEncoder(PlainConvEncoder):
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips, nonlin_first):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first)
        
class FusionUNetDecoder(UNetDecoder):
    def __init__(self, encoder, num_classes, n_conv_per_stage, deep_supervision, nonlin_first = False, norm_op = None, norm_op_kwargs = None, dropout_op = None, dropout_op_kwargs = None, nonlin = None, nonlin_kwargs = None, conv_bias = None):
        super().__init__(encoder, num_classes, n_conv_per_stage, deep_supervision, nonlin_first, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, conv_bias)
        # self.basic_dims = 4
        # self.fusion1 = nn.Conv3d(self.basic_dims * 2 * 8, self.basic_dims * 8, kernel_size = 1, stride = 1, padding = 0, bias = True)
        # self.fusion2 = nn.Conv3d(self.basic_dims * 2 * 16, self.basic_dims * 16, kernel_size = 1, stride = 1, padding = 0, bias = True)
        # self.fusion3 = nn.Conv3d(self.basic_dims * 2 * 32, self.basic_dims * 32, kernel_size = 1, stride = 1, padding = 0, bias = True)
        # self.fusion4 = nn.Conv3d(self.basic_dims * 2 * 64, self.basic_dims * 64, kernel_size = 1, stride = 1, padding = 0, bias = True)
        # self.fusion5 = nn.Conv3d(self.basic_dims * 2 * 80, self.basic_dims * 80, kernel_size = 1, stride = 1, padding = 0, bias = True)
        # self.fusion6 = nn.Conv3d(self.basic_dims * 2 * 80, self.basic_dims * 80, kernel_size = 1, stride = 1, padding = 0, bias = True)
        # self.fusion = nn.Sequential(self.fusion1, self.fusion2, self.fusion3, self.fusion4, self.fusion5, self.fusion6)
        # self.fusion1 = GatedResidualFusion(shared_feature_channels = 32, independent_feature_channels = 16, num_modalities = 2)
        # self.fusion2 = GatedResidualFusion(shared_feature_channels = 64, independent_feature_channels = 32, num_modalities = 2)
        # self.fusion3 = GatedResidualFusion(shared_feature_channels = 128, independent_feature_channels = 64, num_modalities = 2)
        # self.fusion4 = GatedResidualFusion(shared_feature_channels = 256, independent_feature_channels = 128, num_modalities = 2)
        # self.fusion5 = GatedResidualFusion(shared_feature_channels = 320, independent_feature_channels = 160, num_modalities = 2)
        # self.fusion6 = GatedResidualFusion(shared_feature_channels = 320, independent_feature_channels = 160, num_modalities = 2)
        # self.fusion = nn.Sequential(self.fusion1, self.fusion2, self.fusion3, self.fusion4, self.fusion5, self.fusion6)
    # def forward(self, skips, sup_features):
    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            # i = len(self.stages) - s
            # lres_input = self.fusion[i](torch.cat([lres_input, sup_features[i]], dim = 1))
            # block = sup_features[i].shape[1] // 2
            # lres_input = self.fusion[i](lres_input, torch.split(sup_features[i], block, dim = 1))
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r
    
class ResidualFusionBlock(nn.Module):
    def __init__(self, channels, reduction = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv3d(hidden, channels, kernel_size = 1),
            nn.Sigmoid()
        )
    
    def forward(self, f_shared, f_ind):
        delta = f_ind - f_shared
        alpha = self.gate(delta)
        f_fused = f_shared + alpha * delta

        return f_fused

# class PlainConvUNet(nn.Module):
#     def __init__(self,
#                  input_channels: int,
#                  n_stages: int,
#                  features_per_stage: Union[int, List[int], Tuple[int, ...]],
#                  conv_op: Type[_ConvNd],
#                  kernel_sizes: Union[int, List[int], Tuple[int, ...]],
#                  strides: Union[int, List[int], Tuple[int, ...]],
#                  n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
#                  num_classes: int,
#                  n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
#                  conv_bias: bool = False,
#                  norm_op: Union[None, Type[nn.Module]] = None,
#                  norm_op_kwargs: dict = None,
#                  dropout_op: Union[None, Type[_DropoutNd]] = None,
#                  dropout_op_kwargs: dict = None,
#                  nonlin: Union[None, Type[torch.nn.Module]] = None,
#                  nonlin_kwargs: dict = None,
#                  deep_supervision: bool = False,
#                  nonlin_first: bool = False,
#                  dataset = 'MMWHS'
#                  ):
#         """
#         nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
#         """
#         super().__init__()
#         if isinstance(n_conv_per_stage, int):
#             n_conv_per_stage = [n_conv_per_stage] * n_stages
#         if isinstance(n_conv_per_stage_decoder, int):
#             n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
#         assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
#                                                   f"resolution stages. here: {n_stages}. " \
#                                                   f"n_conv_per_stage: {n_conv_per_stage}"
#         assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
#                                                                 f"as we have resolution stages. here: {n_stages} " \
#                                                                 f"stages, so it should have {n_stages - 1} entries. " \
#                                                                 f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
#         self.dataset = dataset
#         self.encoder = FusionUNetEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
#                                         n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
#                                         dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
#                                         nonlin_first=nonlin_first)
#         self.decoder = FusionUNetDecoder(
#             encoder = self.encoder, num_classes = num_classes, n_conv_per_stage = n_conv_per_stage_decoder, deep_supervision = deep_supervision,
#             nonlin_first = False, norm_op = norm_op, norm_op_kwargs = norm_op_kwargs, dropout_op = dropout_op, dropout_op_kwargs = dropout_op_kwargs,
#             nonlin = nonlin, nonlin_kwargs = nonlin_kwargs, conv_bias = conv_bias)
        
#         if self.dataset == 'MMWHS':
#             self.multi_modal_encoder = MultiModalEncoder_MMWHS()
#             self.shared_decoder = SharedDecoder(basic_dims = 8)

#         elif self.dataset == 'Brats':
#             self.multi_modal_encoder = MultiModalEncoder_Brats()
#             self.shared_decoder = SharedDecoder(basic_dims = 4)
#         # self.gated_fusion = MultilevelGatedFusion(channel_list = [32, 64, 128, 256, 320, 320])
#         # self.conv1x1_fusion = nn.Sequential(
#         #     nn.Conv3d(32, 32, kernel_size = 1),
#         #     nn.Conv3d(64, 64, kernel_size = 1),
#         #     nn.Conv3d(128, 128, kernel_size = 1),
#         #     nn.Conv3d(256, 256, kernel_size = 1),
#         #     nn.Conv3d(320, 320, kernel_size = 1),
#         #     nn.Conv3d(320, 320, kernel_size = 1)
#         # )

#     def forward(self, x):
#         skips = self.encoder(x)
#         sup_features = self.multi_modal_encoder(x)

#         if self.dataset == 'MMWHS':
#             ct_img = self.shared_decoder(skips, *sup_features[0])
#             mri_img = self.shared_decoder(skips, *sup_features[1])
#             pred_imgs = torch.cat((ct_img, mri_img), dim = 1)
#             # sup_features = [torch.cat((sup_features[0][i], sup_features[1][i]), dim = 1) for i in range(len(sup_features[0]))]
#         elif self.dataset == 'Brats':
#             t1_img = self.shared_decoder(*sup_features[0])
#             t1ce_img = self.shared_decoder(*sup_features[1])
#             t2_img = self.shared_decoder(*sup_features[2])
#             flair_img = self.shared_decoder(*sup_features[3])
#             pred_imgs = torch.cat((t1_img, t1ce_img, t2_img, flair_img), dim = 1)
#             # sup_features = [torch.cat((sup_features[0][i], sup_features[1][i], sup_features[2][i], sup_features[3][i]), dim = 1) for i in range(len(sup_features[0]))]

#         # sup_features = [self.conv1x1_fusion[i](sup_feat) for i, sup_feat in enumerate(sup_features)]
#         # skips = self.gated_fusion(skips, sup_features)
#         # return self.decoder(skips, sup_features), pred_imgs, (skips, sup_features)
#         return self.decoder(skips), pred_imgs, (skips, sup_features)

#     def compute_conv_feature_map_size(self, input_size):
#         assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
#                                                             "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
#                                                             "Give input_size=(x, y(, z))!"
#         return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

#     @staticmethod
#     def initialize(module):
#         InitWeights_He(1e-2)(module)

class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 dataset = 'MMWHS'
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.dataset = dataset
        self.n_stages = n_stages
        self.shared_encoder = FusionUNetEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = FusionUNetDecoder(
            encoder = self.shared_encoder, num_classes = num_classes, n_conv_per_stage = n_conv_per_stage_decoder, deep_supervision = deep_supervision,
            nonlin_first = False, norm_op = norm_op, norm_op_kwargs = norm_op_kwargs, dropout_op = dropout_op, dropout_op_kwargs = dropout_op_kwargs,
            nonlin = nonlin, nonlin_kwargs = nonlin_kwargs, conv_bias = conv_bias)
        
        if self.dataset == 'MMWHS':
            self.discriminant_encoder = MultiModalEncoder_MMWHS()
            self.recon_decoder = SharedDecoder(basic_dims = 8)

        elif self.dataset == 'BRATS':
            self.discriminant_encoder = MultiModalEncoder_Brats()
            self.recon_decoder = SharedDecoder(basic_dims = 4)

        self.residual_fusion = nn.Sequential(
            ResidualFusionBlock(channels = 32, reduction = 4),
            ResidualFusionBlock(channels = 64, reduction = 4),
            ResidualFusionBlock(channels = 128, reduction = 4),
            ResidualFusionBlock(channels = 256, reduction = 4),
            ResidualFusionBlock(channels = 320, reduction = 4),
            ResidualFusionBlock(channels = 320, reduction = 4)
        )

        self.conv1x1_fusion = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size = 1),
            nn.Conv3d(64, 64, kernel_size = 1),
            nn.Conv3d(128, 128, kernel_size = 1),
            nn.Conv3d(256, 256, kernel_size = 1),
            nn.Conv3d(320, 320, kernel_size = 1),
            nn.Conv3d(320, 320, kernel_size = 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        shared_features = self.shared_encoder(x)
        discriminant_features = self.discriminant_encoder(x)
        # sup_features_ = split_tensor_list_by_modal(sup_features, modal_num = self.modal_num)
        if self.dataset == 'MMWHS':
            ct_img = self.recon_decoder(shared_features, *discriminant_features[0])
            mri_img = self.recon_decoder(shared_features, *discriminant_features[1])
            pred_imgs = torch.cat((ct_img, mri_img), dim = 1)

            discriminant_features = [self.conv1x1_fusion[i](torch.cat([discriminant_features[0][i], discriminant_features[1][i]], dim = 1)) for i in range(self.n_stages)]
            # sup_features = [torch.cat((sup_features[0][i], sup_features[1][i]), dim = 1) for i in range(len(sup_features[0]))]
        elif self.dataset == 'BRATS':
            t1_img = self.recon_decoder(shared_features, *discriminant_features[0])
            t1ce_img = self.recon_decoder(shared_features, *discriminant_features[1])
            t2_img = self.recon_decoder(shared_features, *discriminant_features[2])
            flair_img = self.recon_decoder(shared_features, *discriminant_features[3])
            pred_imgs = torch.cat((t1_img, t1ce_img, t2_img, flair_img), dim = 1)

            discriminant_features = [self.conv1x1_fusion[i](torch.cat([discriminant_features[0][i], discriminant_features[1][i], discriminant_features[2][i], discriminant_features[3][i]], dim = 1)) for i in range(self.n_stages)]

        c_pool_shared_features = [F.adaptive_avg_pool3d(shared_features[i], (1, 1, 1)).squeeze(dim = (2, 3, 4)) for i in range(self.n_stages)]
        c_pool_discriminant_features = [F.adaptive_avg_pool3d(discriminant_features[i], (1, 1, 1)).squeeze(dim = (2, 3, 4)) for i in range(self.n_stages)]

        s_pool_shared_features = [torch.mean(shared_features[i], dim = 1, keepdim = True) for i in range(self.n_stages)]
        s_pool_discriminant_features = [torch.mean(discriminant_features[i], dim = 1, keepdim = True) for i in range(self.n_stages)]

        if self.training:
            # 通道层面对每层的两类特征进行正交损失计算，协方差矩阵Frobenius范数作为正交损失
            c_pool_shared_features_centered = [c_pool_shared_features[i] - c_pool_shared_features[i].mean(dim = 1, keepdim = True) for i in range(self.n_stages)]
            c_pool_discriminant_features_centered = [c_pool_discriminant_features[i] - c_pool_discriminant_features[i].mean(dim = 1, keepdim = True) for i in range(self.n_stages)]
            cov_matrix = [torch.matmul(c_pool_shared_features_centered[i].transpose(0, 1), c_pool_discriminant_features_centered[i] / batch_size - 1) for i in range(self.n_stages)]
            channel_ortho_loss = [torch.norm(cov_matrix[i], p = 'fro') ** 2 for i in range(self.n_stages)]
        
            # 空间层面对每层的两类特征进行正交损失计算
            s_pool_shared_features_centered = [(s_pool_shared_features[i] - s_pool_shared_features[i].mean(dim = (2, 3, 4), keepdim = True)).view(batch_size, -1) for i in range(self.n_stages)]
            s_pool_discriminant_features_centered = [(s_pool_discriminant_features[i] - s_pool_discriminant_features[i].mean(dim = (2, 3, 4), keepdim = True)).view(batch_size, -1) for i in range(self.n_stages)] 
            space_ortho_loss = [F.cosine_similarity(s_pool_shared_features_centered[i], s_pool_discriminant_features_centered[i], dim = 1) ** 2 for i in range(self.n_stages)]

            channel_ortho_loss = torch.mean(torch.stack(channel_ortho_loss))
            space_ortho_loss = torch.mean(torch.stack(space_ortho_loss))

            ortho_loss = channel_ortho_loss + space_ortho_loss
        else:
            ortho_loss = 0.0

        skips = [self.residual_fusion[i](shared_features[i], discriminant_features[i]) for i in range(self.n_stages)]

        return self.decoder(skips), (pred_imgs), ortho_loss

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                               "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)

class ResidualEncoderUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 dataset = 'MMWHS'
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.dataset = dataset
        self.n_stages = n_stages

        self.shared_encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        if self.dataset == 'MMWHS':
            self.discriminant_encoder = MultiModalEncoder_MMWHS()
            self.recon_decoder = SharedDecoder(basic_dims = 8)

        elif self.dataset == 'Brats':
            self.discriminant_encoder = MultiModalEncoder_Brats()
            self.recon_decoder = SharedDecoder(basic_dims = 4)

        self.residual_fusion = nn.Sequential(
            ResidualFusionBlock(channels = 32, reduction = 4),
            ResidualFusionBlock(channels = 64, reduction = 4),
            ResidualFusionBlock(channels = 128, reduction = 4),
            ResidualFusionBlock(channels = 256, reduction = 4),
            ResidualFusionBlock(channels = 320, reduction = 4),
            ResidualFusionBlock(channels = 320, reduction = 4)
        )

        self.conv1x1_fusion = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size = 1),
            nn.Conv3d(64, 64, kernel_size = 1),
            nn.Conv3d(128, 128, kernel_size = 1),
            nn.Conv3d(256, 256, kernel_size = 1),
            nn.Conv3d(320, 320, kernel_size = 1),
            nn.Conv3d(320, 320, kernel_size = 1)
        )
    def forward(self, x):
        batch_size = x.shape[0]
        shared_features = self.shared_encoder(x)
        discriminant_features = self.discriminant_encoders(x)
        # sup_features_ = split_tensor_list_by_modal(sup_features, modal_num = self.modal_num)
        if self.dataset == 'MMWHS':
            ct_img = self.recon_decoder(shared_features, *discriminant_features[0])
            mri_img = self.recon_decoder(shared_features, *discriminant_features[1])
            pred_imgs = torch.cat((ct_img, mri_img), dim = 1)

            discriminant_features = [self.conv1x1_fusion[i](torch.cat([discriminant_features[0][i], discriminant_features[1][i]], dim = 1)) for i in range(self.n_stages)]
            # sup_features = [torch.cat((sup_features[0][i], sup_features[1][i]), dim = 1) for i in range(len(sup_features[0]))]
        elif self.dataset == 'Brats':
            t1_img = self.recon_decoder(shared_features, *discriminant_features[0])
            t1ce_img = self.recon_decoder(shared_features, *discriminant_features[1])
            t2_img = self.recon_decoder(shared_features, *discriminant_features[2])
            flair_img = self.recon_decoder(shared_features, *discriminant_features[3])
            pred_imgs = torch.cat((t1_img, t1ce_img, t2_img, flair_img), dim = 1)

            discriminant_features = [self.conv1x1_fusion[i](torch.cat([discriminant_features[0][i], discriminant_features[1][i], discriminant_features[2][i], discriminant_features[3][i]], dim = 1)) for i in range(self.n_stages)]

        c_pool_shared_features = [F.adaptive_avg_pool3d(shared_features[i], (1, 1, 1)).squeeze(dim = (2, 3, 4)) for i in range(self.n_stages)]
        c_pool_discriminant_features = [F.adaptive_avg_pool3d(discriminant_features[i], (1, 1, 1)).squeeze(dim = (2, 3, 4)) for i in range(self.n_stages)]

        s_pool_shared_features = [torch.mean(shared_features[i], dim = 1, keepdim = True) for i in range(self.n_stages)]
        s_pool_discriminant_features = [torch.mean(discriminant_features[i], dim = 1, keepdim = True) for i in range(self.n_stages)]

        if self.training:
            # 通道层面对每层的两类特征进行正交损失计算，协方差矩阵Frobenius范数作为正交损失
            c_pool_shared_features_centered = [c_pool_shared_features[i] - c_pool_shared_features[i].mean(dim = 1, keepdim = True) for i in range(self.n_stages)]
            c_pool_discriminant_features_centered = [c_pool_discriminant_features[i] - c_pool_discriminant_features[i].mean(dim = 1, keepdim = True) for i in range(self.n_stages)]
            cov_matrix = [torch.matmul(c_pool_shared_features_centered[i].transpose(0, 1), c_pool_discriminant_features_centered[i] / batch_size - 1) for i in range(self.n_stages)]
            channel_ortho_loss = [torch.norm(cov_matrix[i], p = 'fro') ** 2 for i in range(self.n_stages)]
        
            # 空间层面对每层的两类特征进行正交损失计算
            s_pool_shared_features_centered = [(s_pool_shared_features[i] - s_pool_shared_features[i].mean(dim = (2, 3, 4), keepdim = True)).view(batch_size, -1) for i in range(self.n_stages)]
            s_pool_discriminant_features_centered = [(s_pool_discriminant_features[i] - s_pool_discriminant_features[i].mean(dim = (2, 3, 4), keepdim = True)).view(batch_size, -1) for i in range(self.n_stages)] 
            space_ortho_loss = [F.cosine_similarity(s_pool_shared_features_centered[i], s_pool_discriminant_features_centered[i], dim = 1) ** 2 for i in range(self.n_stages)]

            channel_ortho_loss = torch.mean(torch.stack(channel_ortho_loss))
            space_ortho_loss = torch.mean(torch.stack(space_ortho_loss))

            ortho_loss = channel_ortho_loss + space_ortho_loss
        else:
            ortho_loss = 0.0

        skips = [self.residual_fusion[i](shared_features[i], discriminant_features[i]) for i in range(self.n_stages)]

        return self.decoder(skips), (pred_imgs), ortho_loss

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class ResidualUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 dataset = 'MMWHS'
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.input_channels = input_channels
        self.n_stages = n_stages
        self.dataset = dataset

        self.shared_encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)

        self.decoder = UNetResDecoder(self.shared_encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        self.residual_fusion = nn.Sequential(
            ResidualFusionBlock(channels = 32, reduction = 4),
            ResidualFusionBlock(channels = 64, reduction = 4),
            ResidualFusionBlock(channels = 128, reduction = 4),
            ResidualFusionBlock(channels = 256, reduction = 4),
            ResidualFusionBlock(channels = 320, reduction = 4),
            ResidualFusionBlock(channels = 320, reduction = 4)
        )

        self.conv1x1_fusion = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size = 1),
            nn.Conv3d(64, 64, kernel_size = 1),
            nn.Conv3d(128, 128, kernel_size = 1),
            nn.Conv3d(256, 256, kernel_size = 1),
            nn.Conv3d(320, 320, kernel_size = 1),
            nn.Conv3d(320, 320, kernel_size = 1)
        )

        if self.dataset == 'MMWHS':
            self.discriminant_encoders = MultiModalEncoder_MMWHS()
            # self.modal_num = 2
            # self.multi_modal_encoder = MultiModalEncoder(modal_num = 2)
            self.recon_decoder = SharedDecoder(basic_dims = 8)

        elif self.dataset == 'Brats':
            # self.modal_num = 4
            self.discriminant_encoders = MultiModalEncoder_Brats()
            # self.multi_modal_encoder = MultiModalEncoder(modal_num = 4)
            self.recon_decoder = SharedDecoder(basic_dims = 4)

    def forward(self, x):
        batch_size = x.shape[0]
        shared_features = self.shared_encoder(x)
        discriminant_features = self.discriminant_encoders(x)
        # sup_features_ = split_tensor_list_by_modal(sup_features, modal_num = self.modal_num)
        if self.dataset == 'MMWHS':
            ct_img = self.recon_decoder(shared_features, *discriminant_features[0])
            mri_img = self.recon_decoder(shared_features, *discriminant_features[1])
            pred_imgs = torch.cat((ct_img, mri_img), dim = 1)

            discriminant_features = [self.conv1x1_fusion[i](torch.cat([discriminant_features[0][i], discriminant_features[1][i]], dim = 1)) for i in range(self.n_stages)]
            # sup_features = [torch.cat((sup_features[0][i], sup_features[1][i]), dim = 1) for i in range(len(sup_features[0]))]
        elif self.dataset == 'Brats':
            t1_img = self.recon_decoder(shared_features, *discriminant_features[0])
            t1ce_img = self.recon_decoder(shared_features, *discriminant_features[1])
            t2_img = self.recon_decoder(shared_features, *discriminant_features[2])
            flair_img = self.recon_decoder(shared_features, *discriminant_features[3])
            pred_imgs = torch.cat((t1_img, t1ce_img, t2_img, flair_img), dim = 1)

            discriminant_features = [self.conv1x1_fusion[i](torch.cat([discriminant_features[0][i], discriminant_features[1][i], discriminant_features[2][i], discriminant_features[3][i]], dim = 1)) for i in range(self.n_stages)]

        c_pool_shared_features = [F.adaptive_avg_pool3d(shared_features[i], (1, 1, 1)).squeeze(dim = (2, 3, 4)) for i in range(self.n_stages)]
        c_pool_discriminant_features = [F.adaptive_avg_pool3d(discriminant_features[i], (1, 1, 1)).squeeze(dim = (2, 3, 4)) for i in range(self.n_stages)]

        s_pool_shared_features = [torch.mean(shared_features[i], dim = 1, keepdim = True) for i in range(self.n_stages)]
        s_pool_discriminant_features = [torch.mean(discriminant_features[i], dim = 1, keepdim = True) for i in range(self.n_stages)]

        if self.training:
            # 通道层面对每层的两类特征进行正交损失计算，协方差矩阵Frobenius范数作为正交损失
            c_pool_shared_features_centered = [c_pool_shared_features[i] - c_pool_shared_features[i].mean(dim = 1, keepdim = True) for i in range(self.n_stages)]
            c_pool_discriminant_features_centered = [c_pool_discriminant_features[i] - c_pool_discriminant_features[i].mean(dim = 1, keepdim = True) for i in range(self.n_stages)]
            cov_matrix = [torch.matmul(c_pool_shared_features_centered[i].transpose(0, 1), c_pool_discriminant_features_centered[i] / batch_size - 1) for i in range(self.n_stages)]
            channel_ortho_loss = [torch.norm(cov_matrix[i], p = 'fro') ** 2 for i in range(self.n_stages)]
        
            # 空间层面对每层的两类特征进行正交损失计算
            s_pool_shared_features_centered = [(s_pool_shared_features[i] - s_pool_shared_features[i].mean(dim = (2, 3, 4), keepdim = True)).view(batch_size, -1) for i in range(self.n_stages)]
            s_pool_discriminant_features_centered = [(s_pool_discriminant_features[i] - s_pool_discriminant_features[i].mean(dim = (2, 3, 4), keepdim = True)).view(batch_size, -1) for i in range(self.n_stages)] 
            space_ortho_loss = [F.cosine_similarity(s_pool_shared_features_centered[i], s_pool_discriminant_features_centered[i], dim = 1) ** 2 for i in range(self.n_stages)]

            channel_ortho_loss = torch.mean(torch.stack(channel_ortho_loss))
            space_ortho_loss = torch.mean(torch.stack(space_ortho_loss))

            ortho_loss = channel_ortho_loss + space_ortho_loss
        else:
            ortho_loss = 0.0

        skips = [self.residual_fusion[i](shared_features[i], discriminant_features[i]) for i in range(self.n_stages)]

        return self.decoder(skips), (pred_imgs), ortho_loss

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                               "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


if __name__ == '__main__':
    pass
    # a = torch.randn(2, 4, 128, 128, 128)
    # gct = GCT3D(4)
    # b = gct(a)
    # print(b)
    # extractor = HighFrequencyFeatureExtractor(cutoff_ratio = 0.05)
    # batch_size = 2
    # num_modalities = 2
    # height, width, depth = 128, 128, 128

    # dummy_input = torch.randn(batch_size, num_modalities, height, width, depth, dtype = torch.float32).cuda()
    # print(f'原始张量形状为: {dummy_input.shape}')

    # output = extractor(dummy_input)
    # print(f'处理后的张量形状为: {output.shape}')