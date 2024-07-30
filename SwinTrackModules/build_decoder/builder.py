def build_decoder(drop_path_allocator,
                  dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                  z_shape, x_shape, decoder_type,
                  num_decoders=1):
 

    if decoder_type == 'concatenation_feature_fusion':
        from .concatenated_fusion import build_feature_map_generation_decoder
        return build_feature_map_generation_decoder(drop_path_allocator,
                                                    dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,  
                                                    z_shape, x_shape,
                                                    traditional_positional_encoding_enabled=False,
                                                    untied_position_absolute_enabled=True,
                                                    untied_position_relative_enabled=True,
                                                    num_decoders=num_decoders)
    elif decoder_type == 'searchMask_concatenation_feature_fusion':
        from .concatenated_fusion import build_searchMask_feature_map_generation_decoder
        return build_searchMask_feature_map_generation_decoder(drop_path_allocator,
                                                    dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,  
                                                    z_shape, x_shape,
                                                    traditional_positional_encoding_enabled=False,
                                                    untied_position_absolute_enabled=True,
                                                    untied_position_relative_enabled=True,
                                                    num_decoders=num_decoders)
    elif decoder_type == 'concatenation_feature_fusion_dualPos':
        from .concatenated_fusion import build_dualPos_feature_map_generation_decoder
        return build_dualPos_feature_map_generation_decoder(drop_path_allocator,
                                                    dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,  
                                                    z_shape, x_shape,
                                                    traditional_positional_encoding_enabled=False,
                                                    untied_position_absolute_enabled=True,
                                                    untied_position_relative_enabled=True,
                                                    num_decoders=num_decoders)
    elif decoder_type == 'target_query_decoder':
        from .target_query_decoder import build_target_query_decoder
        return build_target_query_decoder(drop_path_allocator,
                                          dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                          z_shape, x_shape)
    else:
        raise NotImplementedError(decoder_type)
