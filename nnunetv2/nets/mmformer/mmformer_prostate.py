import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
from .layers import general_conv3d_prenorm, fusion_prenorm

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 2
patch_size = (16, 16, 2)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*16, stride=(2, 2, 1), pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')

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

        return x1, x2, x3, x4, x5

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=3):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=3):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=(16, 16, 8), mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims*16, num_cls=2)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims*8, num_cls=2)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims*4, num_cls=2)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims*2, num_cls=2)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims*1, num_cls=2)


    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        # de_x5 = self.d4_c1(self.up2(de_x5))
        de_x5 = self.d4_c1(F.interpolate(de_x5, scale_factor = (2, 2, 1), mode = 'trilinear'))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)


    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


class Model(nn.Module):
    def __init__(self, num_cls=3):
        super(Model, self).__init__()
        self.t2_encoder = Encoder()
        self.adc_encoder = Encoder()

        ########### IntraFormer
        self.t2_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.adc_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        
        self.t2_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
        self.adc_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)

        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size[0] * patch_size[1] * patch_size[2], transformer_basic_dims))
        self.adc_pos = nn.Parameter(torch.zeros(1, patch_size[0] * patch_size[1] * patch_size[2], transformer_basic_dims))

        self.t2_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.adc_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        ########### IntraFormer

        ########### InterFormer
        self.multimodal_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals)
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims*num_modals, basic_dims*16*num_modals, kernel_size=1, padding=0)
        ########### InterFormer

        self.masker = MaskModal()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask):
        #extract feature from different layers
        x = x.permute(0, 1, 3, 4, 2)
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 0:1, :, :, :])
        adc_x1, adc_x2, adc_x3, adc_x4, adc_x5 = self.adc_encoder(x[:, 1:2, :, :, :])

        ########### IntraFormer
        t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        adc_token_x5 = self.adc_encode_conv(adc_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)

        t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)
        adc_intra_token_x5 = self.adc_transformer(adc_token_x5, self.adc_pos)

        t2_intra_x5 = t2_intra_token_x5.view(x.size(0), patch_size[0], patch_size[1], patch_size[2], transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        adc_intra_x5 = adc_intra_token_x5.view(x.size(0), patch_size[0], patch_size[1], patch_size[2], transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()

        if self.is_training:
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4,t2_x5)
            adc_pred = self.decoder_sep(adc_x1, adc_x2, adc_x3, adc_x4, adc_x5)
        ########### IntraFormer

        x1 = self.masker(torch.stack((t2_x1, adc_x1), dim=1), mask) #Bx4xCxHWZ
        x2 = self.masker(torch.stack((t2_x2, adc_x2), dim=1), mask)
        x3 = self.masker(torch.stack((t2_x3, adc_x3), dim=1), mask)
        x4 = self.masker(torch.stack((t2_x4, adc_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((t2_intra_x5, adc_intra_x5), dim=1), mask)

        ########### InterFormer
        t2_intra_x5, adc_intra_x5 = torch.chunk(x5_intra, num_modals, dim=1) 
        multimodal_token_x5 = torch.cat((t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         adc_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
        multimodal_pos = torch.cat((self.t2_pos, self.adc_pos), dim=1)
        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), patch_size[0], patch_size[1], patch_size[2], transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
        x5_inter = multimodal_inter_x5

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer
        fuse_pred = fuse_pred.permute(0, 1, 4, 2, 3).contiguous()
        
        if self.is_training:
            t2_pred = t2_pred.permute(0, 1, 4, 2, 3).contiguous()
            adc_pred = adc_pred.permute(0, 1, 4, 2, 3).contiguous()
            preds = [pred.permute(0, 1, 4, 2, 3).contiguous() for pred in preds]
        
        if self.is_training:
            return fuse_pred, (t2_pred, adc_pred), preds
        return fuse_pred