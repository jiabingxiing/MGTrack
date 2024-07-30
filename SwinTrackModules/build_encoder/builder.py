def build_encoder(encoder_type, drop_path_allocator,
                  dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                  z_shape, x_shape,
                  num_encoders=4):
    encoder_type = encoder_type

    if encoder_type == 'concatenation_feature_fusion':  
        from .concatenated_fusion.builder import build_concatenated_fusion_encoder
        return build_concatenated_fusion_encoder(
            drop_path_allocator,
            dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
            z_shape, x_shape,
            traditional_positional_encoding_enabled=False,
            untied_position_absolute_enabled=True, untied_position_relative_enabled=True,
            num_encoders=num_encoders
        )
    elif encoder_type == 'concatenation_feature_fusion_dualPos':  
        from .concatenated_fusion.builder import build_concatenated_fusion_encoder_dual_positional_embe
        return build_concatenated_fusion_encoder_dual_positional_embe(
            drop_path_allocator,
            dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
            z_shape, x_shape,
            traditional_positional_encoding_enabled=False,
            untied_position_absolute_enabled=True, untied_position_relative_enabled=True,
            num_encoders=num_encoders
        )
    elif encoder_type == 'searchMask_concatenation_feature_fusion': 
        from .concatenated_fusion.builder import build_searchMask_concatenated_fusion_encoder
        return build_searchMask_concatenated_fusion_encoder(
            drop_path_allocator,
            dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
            z_shape, x_shape,
            traditional_positional_encoding_enabled=False,
            untied_position_absolute_enabled=True, untied_position_relative_enabled=True,
            num_encoders=num_encoders
        )

    elif encoder_type == 'cross_attention_feature_fusion':
        from .cross_attention_fusion.builder import build_cross_attention_based_encoder
        return build_cross_attention_based_encoder(drop_path_allocator,
                                                   dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                                   z_shape, x_shape, num_layers=4)
    else:
        raise NotImplementedError(encoder_type)
