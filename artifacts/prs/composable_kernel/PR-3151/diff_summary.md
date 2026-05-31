# Diff summary

- **files changed:** 14
- **lines:** +236 / -225
- **kernel-ish files:** 14

## Files (by churn)

- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_large_tensor_invoker.hpp`  (+142/-118)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`  (+27/-30)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data_invoker.hpp`  (+26/-27)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_bwd_data_example.inc`  (+8/-8)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_fwd_example.inc`  (+8/-8)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_invoker.hpp`  (+4/-5)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage_invoker.hpp`  (+4/-5)
- `example/ck_tile/20_grouped_convolution/conv_configs.hpp`  (+1/-7)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+4/-3)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data.cpp`  (+3/-3)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward.cpp`  (+3/-3)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_bias_clamp.cpp`  (+3/-3)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_large_tensor.cpp`  (+3/-3)
- `include/ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp`  (+0/-2)

## Key added lines (kernel files)

**`example/ck_tile/20_grouped_convolution/conv_configs.hpp`**
```
static constexpr bool TransposeC = false;
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data.cpp`**
```
template <template <typename PrecType> typename ConvConfig>
ConvConfig<ck_tile::half_t>,
ConvConfig<ck_tile::bf16_t>,
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data_invoker.hpp`**
```
typename ConvConfig,
ck_tile::sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
ck_tile::sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
ck_tile::sequence<ConvConfig::M_Warp_Tile,
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_invoker.hpp`**
```
ck_tile::sequence<ConvConfig::M_Warp_Tile,
ConvConfig::N_Warp_Tile,
ConvConfig::K_Warp_Tile>>;
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage_invoker.hpp`**
```
ck_tile::sequence<ConvConfig::M_Warp_Tile,
ConvConfig::N_Warp_Tile,
ConvConfig::K_Warp_Tile>>;
```
