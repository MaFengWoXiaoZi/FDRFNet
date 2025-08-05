import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.channel_token = channel_token ## whether to use channel as tokens

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out


class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

# class GatedAttention3D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size = 1):
#         super(GatedAttention3D, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size)
#         self.conv2 = nn.Conv3d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x1, x2):
#         x_cat = torch.cat([x1, x2], dim = 1)
#         x = F.relu(self.conv1(x_cat))

#         alpha = self.sigmoid(self.conv2(x))

#         fused_output = alpha * x1 + (1 - alpha) * x2

#         return fused_output

class GCT3D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT3D, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3, 4), keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3, 4), keepdim=True) * self.alpha
            norm = self.gamma / \
                (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate
    
class GateAttention3D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GateAttention3D, self).__init__()

        self.GCT1 = GCT3D(num_channels, epsilon, mode, after_relu)
        self.GCT2 = GCT3D(num_channels, epsilon, mode, after_relu)

    def forward(self, x1, x2):
        x1 = self.GCT1(x1)
        x2 = self.GCT2(x2)

        return x1, x2

    
class ResidualMambaEncoder(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i // j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True
            

        print(f"feature_map_sizes: {feature_map_sizes}")
        print(f"do_channel_token: {do_channel_token}")

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op = conv_op,
                input_channels = input_channels,
                output_channels = stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ), 
            *[
                BasicBlockD(
                    conv_op = conv_op,
                    input_channels = stem_channels,
                    output_channels = stem_channels,
                    kernel_size = kernel_sizes[0],
                    stride = 1,
                    conv_bias = conv_bias,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )

        input_channels = stem_channels

        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op = conv_op,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    input_channels = input_channels,
                    output_channels = features_per_stage[s],
                    kernel_size = kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op = conv_op,
                        input_channels = features_per_stage[s],
                        output_channels = features_per_stage[s],
                        kernel_size = kernel_sizes[s],
                        stride = 1,
                        conv_bias = conv_bias,
                        norm_op = norm_op,
                        norm_op_kwargs = norm_op_kwargs,
                        nonlin = nonlin,
                        nonlin_kwargs = nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )

            mamba_layers.append(
                MambaLayer(
                    dim = np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                    channel_token = do_channel_token[s]
                )
            )

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        #self.dropout_op = dropout_op
        #self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
    
        self.ga1 = GateAttention3D(32)
        self.fusion1 = nn.Conv3d(in_channels = 64, out_channels = 32, kernel_size = 1)
        self.ga2 = GateAttention3D(64)
        self.fusion2 = nn.Conv3d(in_channels = 128, out_channels = 64, kernel_size = 1)
        self.ga3 = GateAttention3D(128)
        self.fusion3 = nn.Conv3d(in_channels = 256, out_channels = 128, kernel_size = 1)
        self.ga4 = GateAttention3D(256)
        self.fusion4 = nn.Conv3d(in_channels = 512, out_channels = 256, kernel_size = 1)
        self.ga5 = GateAttention3D(320)
        self.fusion5 = nn.Conv3d(in_channels = 640, out_channels = 320, kernel_size = 1)
        self.ga6 = GateAttention3D(320)
        self.fusion6 = nn.Conv3d(in_channels = 640, out_channels = 320, kernel_size = 1)

        self.gated_attention = nn.Sequential(*[
            self.ga1, self.ga2, self.ga3, self.ga4, self.ga5, self.ga6
        ])
        self.fusion = nn.Sequential(*[
            self.fusion1, self.fusion2, self.fusion3, self.fusion4, self.fusion5, self.fusion6
        ])

    def forward(self, x, sup_features):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            # cur_sup_features = [sup_features[i][s] for i in range(len(sup_features))]
            # cur_sup_features = torch.cat(cur_sup_features, dim = 1)
            # x = torch.cat([x, cur_sup_features], dim = 1)
            # x = self.fusion[s](x)
            x = self.stages[s](x)
            cur_sup_features = [sup_features[i][s] for i in range(len(sup_features))]
            cur_sup_features = torch.cat(cur_sup_features, dim = 1)
            # x = torch.cat([x, cur_sup_features], dim = 1)
            x, cur_sup_features = self.gated_attention[s](x, cur_sup_features)
            x = self.fusion[s](torch.cat([x, cur_sup_features], dim = 1))
            x = self.mamba_layers[s](x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output

class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        stages = []

        upsample_layers = []

        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op = encoder.conv_op,
                input_channels = input_features_below,
                output_channels = input_features_skip,
                pool_op_kernel_size = stride_for_upsampling,
                mode='nearest'
            ))

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op = encoder.conv_op,
                    norm_op = encoder.norm_op,
                    norm_op_kwargs = encoder.norm_op_kwargs,
                    nonlin = encoder.nonlin,
                    nonlin_kwargs = encoder.nonlin_kwargs,
                    input_channels = 2 * input_features_skip,
                    output_channels = input_features_skip,
                    kernel_size = encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[
                    BasicBlockD(
                        conv_op = encoder.conv_op,
                        input_channels = input_features_skip,
                        output_channels = input_features_skip,
                        kernel_size = encoder.kernel_sizes[-(s + 1)],
                        stride = 1,
                        conv_bias = encoder.conv_bias,
                        norm_op = encoder.norm_op,
                        norm_op_kwargs = encoder.norm_op_kwargs,
                        nonlin = encoder.nonlin,
                        nonlin_kwargs = encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s-1] - 1)
                ]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

        # self.fusion1 = nn.Conv3d(in_channels = 64, out_channels = 32, kernel_size = 1)
        # self.fusion2 = nn.Conv3d(in_channels = 128, out_channels = 64, kernel_size = 1)
        # self.fusion3 = nn.Conv3d(in_channels = 256, out_channels = 128, kernel_size = 1)
        # self.fusion4 = nn.Conv3d(in_channels = 512, out_channels = 256, kernel_size = 1)
        # self.fusion5 = nn.Conv3d(in_channels = 640, out_channels = 320, kernel_size = 1)
        # self.fusion6 = nn.Conv3d(in_channels = 640, out_channels = 320, kernel_size = 1)
        # self.fusion = [self.fusion1, self.fusion2, self.fusion3, self.fusion4, self.fusion5, self.fusion6]

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    # def forward(self, skips, sup_features):
    #     lres_input = skips[-1]
    #     seg_outputs = []
    #     for s in range(len(self.stages)):
    #         cur_sup_features = [sup_features[i][-(s+1)] for i in range(len(sup_features))]
    #         cur_sup_features = torch.cat(cur_sup_features, dim = 1)
    #         lres_input = torch.cat([lres_input, cur_sup_features], dim = 1)
    #         lres_input = self.fusion[-(s+1)](lres_input)

    #         x = self.upsample_layers[s](lres_input)
    #         x = torch.cat((x, skips[-(s+2)]), 1)
    #         x = self.stages[s](x)
    #         if self.deep_supervision:
    #             # if s == len(self.stages) - 1:
    #             #     cur_sup_features = [sup_features[i][-(s+2)] for i in range(len(sup_features))]
    #             #     cur_sup_features = torch.cat(cur_sup_features, dim = 1)
    #             #     x = torch.cat([x, cur_sup_features], dim = 1)
    #             #     x = self.fusion[-(s+2)](x)
    #             seg_outputs.append(self.seg_layers[s](x))
    #         elif s == (len(self.stages) - 1):
    #             seg_outputs.append(self.seg_layers[-1](x))
    #         lres_input = x

    #     seg_outputs = seg_outputs[::-1]

    #     if not self.deep_supervision:
    #         r = seg_outputs[0]
    #     else:
    #         r = seg_outputs
    #     return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output
    
class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size = 3, stride = 1, padding = 1, pad_type = 'zeros', norm = 'in', is_training = True, act_type = 'lrelu', relufactor = 0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels = in_ch, out_channels = out_ch, kernel_size = k_size, stride = stride, padding = padding, padding_mode = pad_type, bias = True)
        
        self.norm = nn.BatchNorm3d(in_ch)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope = relufactor, inplace = True)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
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
    
class MultiModalEncoder(nn.Module):
    def __init__(self, modal_num = 4):
        super().__init__()
        self.modal_num = modal_num
        self.t1_encoder = SingalModalEncoder()
        self.t1ce_encoder = SingalModalEncoder()
        self.t2_encoder = SingalModalEncoder()
        self.flair_encoder = SingalModalEncoder()

    def forward(self, x):
        t1_features = self.t1_encoder(x[:, 0, ...].unsqueeze(1))
        t1ce_features = self.t1ce_encoder(x[:, 0, ...].unsqueeze(1))
        t2_features = self.t2_encoder(x[:, 0, ...].unsqueeze(1))
        flair_features = self.flair_encoder(x[:, 0, ...].unsqueeze(1))

        return t1_features, t1ce_features, t2_features, flair_features       

class SharedDecoder(nn.Module):
    def __init__(self, basic_dims = 4):
        super().__init__()

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

    def forward(self, x1, x2, x3, x4, x5, x6):
        de_x6 = self.d5_c1(self.d5(x6))

        cat_x5 = torch.cat((de_x6, x5), dim = 1)
        de_x5 = self.d5_out(self.d5_c2(cat_x5))
        de_x5 = self.d4_c1(self.d4(de_x5))

        cat_x4 = torch.cat((de_x5, x4), dim = 1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim = 1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim = 1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim = 1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        predicted_img = self.conv3d(de_x1)

        return predicted_img


class UMambaEnc(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
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
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1    

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1


        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualMambaEncoder(
            input_size,
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels
        )

        # my module
        self.multi_modal_encoder = MultiModalEncoder(modal_num = 4)
        self.shared_decoder = SharedDecoder(basic_dims = 4)

        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        sup_features = self.multi_modal_encoder(x)

        t1_img = self.shared_decoder(*sup_features[0])
        t1ce_img = self.shared_decoder(*sup_features[1])
        t2_img = self.shared_decoder(*sup_features[2])
        flair_img = self.shared_decoder(*sup_features[3])

        pred_img = torch.cat((t1_img, t1ce_img, t2_img, flair_img), dim = 1)

        # skips = self.encoder(x)
        skips = self.encoder(x, sup_features)

        seg = self.decoder(skips)

        # return self.decoder(skips, sup_features)
        return pred_img, seg

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def get_umamba_enc_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
    ):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaEnc'
    network_class = UMambaEnc
    kwargs = {
        'UMambaEnc': {
            'input_size': configuration_manager.patch_size,
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))

    return model

if __name__ == '__main__':
    a = torch.randn(2, 4, 128, 128, 128)
    # model = MultiModalEncoder()
    # b = model(a)
    # print(b)
    gct = GCT3D(4)
    b = gct(a)
    print(b)