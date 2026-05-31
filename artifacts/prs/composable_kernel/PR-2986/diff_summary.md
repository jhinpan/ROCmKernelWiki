# Diff summary

- **files changed:** 17
- **lines:** +764 / -278
- **kernel-ish files:** 17

## Files (by churn)

- `include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_weight_to_gemm.hpp`  (+538/-138)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+89/-25)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_invoker.hpp`  (+31/-30)
- `example/ck_tile/20_grouped_convolution/conv_configs.hpp`  (+41/-10)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage_invoker.hpp`  (+25/-25)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_bwd_weight_example.inc`  (+8/-8)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_utils.hpp`  (+1/-14)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage.cpp`  (+6/-6)
- `include/ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp`  (+6/-5)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight.cpp`  (+5/-5)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`  (+5/-3)
- `include/ck_tile/core/algorithm/static_encoding_pattern.hpp`  (+2/-3)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data.cpp`  (+2/-2)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward.cpp`  (+2/-2)
- `include/ck_tile/host/check_err.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/20_grouped_convolution/conv_configs.hpp`**
```
struct ConvConfigBase
static constexpr ck_tile::index_t VectorSizeA = 4;
static constexpr ck_tile::index_t VectorSizeB = 8;
static constexpr ck_tile::index_t VectorSizeC = 8;
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data.cpp`**
```
return !run_grouped_conv_bwd_data_example<ConvConfigComputeV3_WMMA>(argc, argv);
return !run_grouped_conv_bwd_data_example<ConvConfigComputeV3>(argc, argv);
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight.cpp`**
```
template <template <typename PrecType> typename ConvConfig>
ConvConfig<ck_tile::half_t>,
ConvConfig<ck_tile::bf16_t>,
return !run_grouped_conv_bwd_weight_example<ConvConfigComputeV3_WMMA>(arg_parser);
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_invoker.hpp`**
```
typename ConvConfig,
ck_tile::sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
ck_tile::sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
sequence<ConvConfig::M_Warp_Tile, ConvConfig::N_Warp_Tile, ConvConfig::K_Warp_Tile>,
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage.cpp`**
```
template <template <typename PrecType> typename ConvConfig>
ConvConfig<ck_tile::half_t>,
ConvConfig<ck_tile::bf16_t>,
return !run_grouped_conv_bwd_weight_example<ConvConfigComputeV3_WMMA>(arg_parser);
```
