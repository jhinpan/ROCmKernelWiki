# Diff summary

- **files changed:** 19
- **lines:** +1057 / -564
- **kernel-ish files:** 19

## Files (by churn)

- `example/ck_tile/20_grouped_convolution/gemm_configs.hpp`  (+303/-0)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data_invoker.hpp`  (+151/-102)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`  (+149/-91)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_invoker.hpp`  (+118/-70)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage_invoker.hpp`  (+114/-62)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+68/-76)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp`  (+41/-50)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+42/-42)
- `include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_weight_to_gemm.hpp`  (+8/-11)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_bwd_data_example.inc`  (+8/-8)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_bwd_weight_example.inc`  (+8/-8)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_fwd_example.inc`  (+8/-8)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward.cpp`  (+9/-5)
- `include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_data_to_gemm.hpp`  (+7/-7)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage.cpp`  (+6/-5)

## Key added lines (kernel files)

**`example/ck_tile/20_grouped_convolution/gemm_configs.hpp`**
```
struct GemmConfigBase
static constexpr bool kPadM = true;
static constexpr bool kPadN = true;
static constexpr bool kPadK = true;
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data.cpp`**
```
template <template <typename PrecType> typename GemmConfig>
GemmConfig<ck_tile::half_t>,
GemmConfig<ck_tile::bf16_t>,
return !run_grouped_conv_bwd_data_example<GemmConfigComputeV3_WMMA>(argc, argv);
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data_invoker.hpp`**
```
typename GemmConfig,
using GemmShape = ck_tile::TileGemmShape<
ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight.cpp`**
```
template <template <typename PrecType> typename GemmConfig>
GemmConfig<ck_tile::half_t>,
GemmConfig<ck_tile::bf16_t>,
return !run_grouped_conv_bwd_weight_example<GemmConfigComputeV3_WMMA>(arg_parser);
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_invoker.hpp`**
```
typename GemmConfig,
using GemmShape = ck_tile::TileGemmShape<
ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
```
