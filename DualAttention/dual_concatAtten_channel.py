import torch.nn as nn
from DualAttention.dual_attention import LayerNorm, MDTA, FourierUnit, GatedFeedForward
import torch
from SwinTrackModules.build_encoder.builder import build_encoder
from SwinTrackModules.drop_path import DropPathAllocator
from DualAttention.dual_attention_with_matrix import Adaptive_Channel_Attention, only_Channel_Attention, Channel_Fourier_Interaction, Channel_Fourier_Interaction_withResAndMLP



class channelAttn_dwConv_block(nn.Module):
    def __init__(self, dim):
        super(channelAttn_dwConv_block, self).__init__()
        self.conv2_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm2_1 = LayerNorm(dim, data_format="channels_first")
        self.dw_conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.transposed_attn = MDTA(dim, 16)
        self.conv_fuse2 = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.weight_generate_2 = FourierUnit(dim, dim, groups=1)
        self.conv1_2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm2_2 = LayerNorm(dim, data_format="channels_first")
        self.ffn2 = GatedFeedForward(dim)

    def forward(self, x):
        shortcut2 = x  
        x = self.norm2_1(x)
        x = self.conv2_1(x)
        conv_2 = self.dw_conv2(x)  
        trans_2 = self.transposed_attn(x)  

        combined = self.conv_fuse2(torch.cat([conv_2, trans_2], dim=1))  
        w2 = self.weight_generate_2(combined)  

        nconv_2 = conv_2 * w2
        ntrans_2 = trans_2 * (1 - w2)

        x = shortcut2 + self.conv1_2(nconv_2 + ntrans_2)  
        x = x + self.ffn2(self.norm2_2(x))  

        return x


class simple_channelAttn_dwConv_block(nn.Module):
    def __init__(self, dim):
        super(simple_channelAttn_dwConv_block, self).__init__()
        self.norm2_1 = LayerNorm(dim, data_format="channels_first")
        self.dw_conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.transposed_attn = MDTA(dim, 16)
        self.weight_generate_2 = FourierUnit(dim, dim, groups=1)
        self.norm2_2 = LayerNorm(dim, data_format="channels_first")
        self.ffn2 = GatedFeedForward(dim)
        self.norm3_2 = LayerNorm(dim, data_format="channels_first")

    def forward(self, x):
        shortcut2 = x  
        x = self.norm2_1(x)
        # x = self.conv2_1(x)
        conv_2 = self.dw_conv2(x) 
        trans_2 = self.transposed_attn(x)  

        combined = self.norm3_2(conv_2 + trans_2)  

        w2 = self.weight_generate_2(combined)  

        nconv_2 = conv_2 * w2
        ntrans_2 = trans_2 * (1 - w2)

        x = shortcut2 + nconv_2 + ntrans_2  
        x = x + self.ffn2(self.norm2_2(x))  

        return x
# ==========================================================================================================================================================================


class concat_spatial_attn_block(nn.Module):
    def __init__(self, dim=384, num_heads=8, mlp_ratio=4, qkv_bias=True, drop_rate=0, attn_drop_rate=0,
                        encoder_type='concatenation_feature_fusion', num_encoders=1, z_shape=None, x_shape=None
                 ):
        super().__init__()
        if x_shape is None:
            self.x_shape = [20, 20]
        if z_shape is None:
            self.z_shape = [20, 20]
        drop_path_allocator = DropPathAllocator(0.1)
        with drop_path_allocator:
            self.concat_encoder = build_encoder(
                drop_path_allocator=drop_path_allocator,
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate, z_shape=[z_shape, z_shape], x_shape=[x_shape, x_shape],
                encoder_type=encoder_type, num_encoders=num_encoders
            )

    def forward(self, tem_mask_mem, tem_fpnout_feat, search_fpnout_feat, z_pos=None, x_pos=None, frame_id=0, attn_layer_num=0):
        tem_out, search_out = self.concat_encoder(tem_mask_mem, tem_fpnout_feat, search_fpnout_feat, z_pos=z_pos, x_pos=x_pos, frame_id=frame_id, attn_layer_num=attn_layer_num)  # (5,400,384), (5,400,384), (5,400,384),--> (5,400,384), (5,400,384)
        return tem_out, search_out


# ======================================================================================================================
class dual2_spatialFrequencyInteraction_deepSeek(nn.Module):
    def __init__(self, dim=384, num_heads=8, mlp_ratio=4, qkv_bias=True, drop_rate=0, attn_drop_rate=0,
                 encoder_type='concatenation_feature_fusion', num_encoders=1, z_shape=None, x_shape=None):
        super(dual2_spatialFrequencyInteraction_deepSeek, self).__init__()
        self.layers = nn.ModuleList([
            self._create_layer(dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, encoder_type, num_encoders, z_shape, x_shape)
            for _ in range(2)
        ])

    def _create_layer(self, dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, encoder_type, num_encoders, z_shape, x_shape):
        return nn.ModuleDict({
            'concat_spatial_attn': concat_spatial_attn_block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate, encoder_type=encoder_type, num_encoders=num_encoders, z_shape=z_shape, x_shape=x_shape
            ),
            'frequencyAttn_tem': FourierUnit(in_channels=dim, out_channels=dim),
            'frequencyAttn_search': FourierUnit(in_channels=dim, out_channels=dim),
            'frequencyInteraction_tem': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 8, kernel_size=1),
                nn.BatchNorm2d(dim // 8),
                nn.GELU(),
                nn.Conv2d(dim // 8, dim, kernel_size=1),
            ),
            'frequencyInteraction_search': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 8, kernel_size=1),
                nn.BatchNorm2d(dim // 8),
                nn.GELU(),
                nn.Conv2d(dim // 8, dim, kernel_size=1),
            ),
            'spatialInteraction_tem': nn.Sequential(
                nn.Conv2d(dim, dim // 16, kernel_size=1),
                nn.BatchNorm2d(dim // 16),
                nn.GELU(),
                nn.Conv2d(dim // 16, 1, kernel_size=1)
            ),
            'spatialInteraction_search': nn.Sequential(
                nn.Conv2d(dim, dim // 16, kernel_size=1),
                nn.BatchNorm2d(dim // 16),
                nn.GELU(),
                nn.Conv2d(dim // 16, 1, kernel_size=1)
            ),
            'proj_tem': nn.Linear(dim, dim),
            'proj_search': nn.Linear(dim, dim)
        })

    def forward(self, tem_mask_mem, tem_fpnout_feat, search_fpnout_feat, z_pos=None, x_pos=None):
        b, dim, h_t, w_t = tem_fpnout_feat.size()
        _, _, h_s, w_s = search_fpnout_feat.size()

        if tem_mask_mem is not None:
            tem_mask_mem = tem_mask_mem.flatten(2).permute(0, 2, 1).contiguous()
        reshape_tem_fpnout_feat = tem_fpnout_feat.flatten(2).permute(0, 2, 1).contiguous()  
        reshape_search_fpnout_feat = search_fpnout_feat.flatten(2).permute(0, 2, 1).contiguous()  

        tem_mix_branch = reshape_tem_fpnout_feat
        search_mix_branch = reshape_search_fpnout_feat

        for layer in self.layers:
            tem_spatialOut, search_spatialOut = layer['concat_spatial_attn'](
                tem_mask_mem, tem_mix_branch, search_mix_branch, z_pos=z_pos, x_pos=x_pos  
            )
            reshaped_back_tem_spatialOut = tem_spatialOut.permute(0, 2, 1).contiguous().reshape(b, dim, h_t, w_t)  
            reshaped_back_search_spatialOut = search_spatialOut.permute(0, 2, 1).contiguous().reshape(b, dim, h_s, w_s)  

            tem_frequencyOut = layer['frequencyAttn_tem'](tem_mix_branch.view(b, dim, h_t, w_t).contiguous())  
            search_frequencyOut = layer['frequencyAttn_search'](search_mix_branch.view(b, dim, h_s, w_s).contiguous())  

            tem_frequency_map = layer['frequencyInteraction_tem'](tem_frequencyOut)  
            search_frequency_map = layer['frequencyInteraction_search'](search_frequencyOut)  
            tem_spatial_map = layer['spatialInteraction_tem'](reshaped_back_tem_spatialOut)  
            search_spatial_map = layer['spatialInteraction_search'](reshaped_back_search_spatialOut)  #

            tem_frequency_branch = tem_frequencyOut * tem_spatial_map  # 
            tem_spatial_branch = reshaped_back_tem_spatialOut * tem_frequency_map  #
            tem_frequency_branch = tem_frequency_branch.flatten(2).permute(0, 2, 1).contiguous()  
            tem_spatial_branch = tem_spatial_branch.flatten(2).permute(0, 2, 1).contiguous()  
            tem_mix_branch = layer['proj_tem'](tem_frequency_branch + tem_spatial_branch) 

            search_frequency_branch = search_frequencyOut * search_spatial_map  
            search_spatial_branch = reshaped_back_search_spatialOut * search_frequency_map  
            search_frequency_branch = search_frequency_branch.flatten(2).permute(0, 2, 1).contiguous()  
            search_spatial_branch = search_spatial_branch.flatten(2).permute(0, 2, 1).contiguous()  # 
            search_mix_branch = layer['proj_search'](search_frequency_branch + search_spatial_branch)  # 

        return tem_mix_branch, search_mix_branch  # 
