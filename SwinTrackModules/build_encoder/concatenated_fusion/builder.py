from SwinTrackModules.self_attention_block import SelfAttentionBlock, searchMaskSelfAttentionBlock
from .concatenated_fusion import ConcatenatedFusion, ConcatenatedFusion_dualPositionEmbed, searchMaskConcatenatedFusion


def build_concatenated_fusion_encoder(drop_path_allocator,
                                      dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                      z_shape, x_shape, traditional_positional_encoding_enabled=False,
                                      untied_position_absolute_enabled=True, untied_position_relative_enabled=True,
                                      num_encoders=4):

    untied_z_pos_enc = None
    untied_x_pos_enc = None
    rpe_index = None
    rpe_bias_table = None


    if untied_position_absolute_enabled:  
        from ...positional_encoding.untied.absolute import Untied2DPositionalEncoder

        untied_z_pos_enc = Untied2DPositionalEncoder(dim, num_heads, z_shape[0], z_shape[1])  
        untied_x_pos_enc = Untied2DPositionalEncoder(dim, num_heads, x_shape[0], x_shape[1])  

    if untied_position_relative_enabled:  
        from ...positional_encoding.untied.relative import RelativePosition2DEncoder, generate_2d_concatenated_self_attention_relative_positional_encoding_index
        rpe_index = generate_2d_concatenated_self_attention_relative_positional_encoding_index((z_shape[1], z_shape[0]), (x_shape[1], x_shape[0]))  
        rpe_bias_table = RelativePosition2DEncoder(num_heads, rpe_index.max() + 1)  

    num_encoders = num_encoders  
    encoder_modules = []
    for index_of_encoder in range(num_encoders):  
        encoder_modules.append(
            SelfAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator.allocate(),
                               attn_pos_encoding_only=not traditional_positional_encoding_enabled)  
        )
        drop_path_allocator.increase_depth()

    encoder = ConcatenatedFusion(encoder_modules, untied_z_pos_enc, untied_x_pos_enc,
                                 rpe_bias_table, rpe_index)
    return encoder
def build_searchMask_concatenated_fusion_encoder(drop_path_allocator,
                                      dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                      z_shape, x_shape, traditional_positional_encoding_enabled=False,
                                      untied_position_absolute_enabled=True, untied_position_relative_enabled=True,
                                      num_encoders=4):

    untied_z_pos_enc = None
    untied_x_pos_enc = None
    rpe_index = None
    rpe_bias_table = None


    if untied_position_absolute_enabled:  
        from ...positional_encoding.untied.absolute import Untied2DPositionalEncoder

        untied_z_pos_enc = Untied2DPositionalEncoder(dim, num_heads, z_shape[0], z_shape[1])  
        untied_x_pos_enc = Untied2DPositionalEncoder(dim, num_heads, x_shape[0], x_shape[1])  

    if untied_position_relative_enabled:  
        from ...positional_encoding.untied.relative import RelativePosition2DEncoder, generate_2d_concatenated_self_attention_relative_positional_encoding_index
        rpe_index = generate_2d_concatenated_self_attention_relative_positional_encoding_index((z_shape[1], z_shape[0]), (x_shape[1], x_shape[0]))  
        rpe_bias_table = RelativePosition2DEncoder(num_heads, rpe_index.max() + 1)  

    num_encoders = num_encoders  
    encoder_modules = []
    for index_of_encoder in range(num_encoders):  
        encoder_modules.append(
            searchMaskSelfAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator.allocate(),
                               attn_pos_encoding_only=not traditional_positional_encoding_enabled)  
        )
        drop_path_allocator.increase_depth()

    encoder = searchMaskConcatenatedFusion(encoder_modules, untied_z_pos_enc, untied_x_pos_enc,
                                 rpe_bias_table, rpe_index)
    return encoder

def build_concatenated_fusion_encoder_dual_positional_embe(drop_path_allocator,
                                      dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                      z_shape, x_shape, traditional_positional_encoding_enabled=False,
                                      untied_position_absolute_enabled=True, untied_position_relative_enabled=True,
                                      num_encoders=4):

    untied_zx_pos_enc = None
    
    rpe_index = None
    rpe_bias_table = None

    

    if untied_position_absolute_enabled:  
        from ...positional_encoding.untied.absolute import dualImag_Untied2DPositionalEncoder

        untied_zx_pos_enc = dualImag_Untied2DPositionalEncoder(dim, num_heads, z_shape[0], z_shape[1], x_shape[0], x_shape[1])  
        
    if untied_position_relative_enabled: 
        from ...positional_encoding.untied.relative import RelativePosition2DEncoder, generate_dualImg_2d_concatenated_self_attention_relative_positional_encoding_index
        rpe_index = generate_dualImg_2d_concatenated_self_attention_relative_positional_encoding_index((z_shape[1], z_shape[0]), (x_shape[1], x_shape[0]))  
        rpe_bias_table = RelativePosition2DEncoder(num_heads, rpe_index.max() + 1)  

    num_encoders = num_encoders  
    encoder_modules = []
    for index_of_encoder in range(num_encoders):  
        encoder_modules.append(
            SelfAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator.allocate(),
                               attn_pos_encoding_only=not traditional_positional_encoding_enabled)  
        )
        drop_path_allocator.increase_depth()

    encoder = ConcatenatedFusion_dualPositionEmbed(encoder_modules, untied_zx_pos_enc,
                                 rpe_bias_table, rpe_index)
    return encoder
