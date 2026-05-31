# Diff summary

- **files changed:** 16
- **lines:** +864 / -361
- **kernel-ish files:** 15

## Files (by churn)

- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight.cpp`  (+26/-191)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage_invoker.hpp`  (+215/-0)
- `include/ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp`  (+111/-36)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_invoker.hpp`  (+145/-0)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_bwd_weight_example.inc`  (+92/-32)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage.cpp`  (+67/-0)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+28/-22)
- `include/ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp`  (+36/-9)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data.cpp`  (+26/-17)
- `include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_data_to_gemm.hpp`  (+32/-9)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward.cpp`  (+24/-15)
- `include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_weight_to_gemm.hpp`  (+28/-10)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp`  (+21/-13)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+9/-6)
- `example/ck_tile/20_grouped_convolution/CMakeLists.txt`  (+3/-0)

## Key added lines (kernel files)

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data.cpp`**
```
constexpr ck_tile::index_t VectorSizeA = 1;
constexpr ck_tile::index_t VectorSizeB = 1;
constexpr auto ConvSpec      = ck_tile::ConvolutionSpecialization::Default;
using TilePartitioner        = ck_tile::GemmTile1DPartitioner<CodegenShape>;
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight.cpp`**
```
int run_grouped_conv_bwd_weight_example(ck_tile::ArgParser& arg_parser)
using Invoker = GroupedConvolutionBackwardWeightInvoker;
return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
GemmWarpConfig,
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_invoker.hpp`**
```
struct GroupedConvolutionBackwardWeightInvoker
template <ck_tile::index_t NDimSpatial,
typename GemmWarpConfig,
typename InDataType,
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage.cpp`**
```
template <typename GemmWarpConfig>
int run_grouped_conv_bwd_weight_example(ck_tile::ArgParser& arg_parser)
using Invoker = GroupedConvolutionBackwardWeightTwoStageInvoker;
std::string data_type  = arg_parser.get_str("prec");
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage_invoker.hpp`**
```
struct GroupedConvolutionBackwardWeightTwoStageInvoker
template <ck_tile::index_t NDimSpatial,
typename GemmWarpConfig,
typename InDataType,
```
