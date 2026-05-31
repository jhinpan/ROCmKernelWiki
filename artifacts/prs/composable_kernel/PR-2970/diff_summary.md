# Diff summary

- **files changed:** 8
- **lines:** +1133 / -315
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+303/-95)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_large_tensor_invoker.hpp`  (+388/-0)
- `include/ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp`  (+177/-132)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`  (+112/-83)
- `include/ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp`  (+82/-0)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_large_tensor.cpp`  (+63/-0)
- `example/ck_tile/20_grouped_convolution/CMakeLists.txt`  (+7/-4)
- `include/ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`**
```
if(s.log_level_ > 0)
std::cout << "[INVOKER] grouped_conv_fwd called, NDimSpatial=" << NDimSpatial << "\n";
const auto Run = [&]<bool EnableSplitImage>(const auto has_hot_loop_,
const auto tail_number_,
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_forward_large_tensor.cpp`**
```
template <template <typename PrecType> typename GemmConfig>
int run_grouped_conv_fwd_example(int argc, char* argv[])
using Invoker = GroupedConvolutionForwardInvoker;
auto [result, arg_parser] = create_args(argc, argv);
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_forward_large_tensor_invoker.hpp`**
```
struct GroupedConvolutionForwardInvoker
template <ck_tile::index_t NDimSpatial,
typename GemmConfig,
typename InDataType,
```

**`include/ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp`**
```
constexpr long_index_t TwoGB          = (long_index_t{1} << 31); // 2GB threshold
```

**`include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`**
```
transformer_ = ConvToGemmFwdTransformer{in_g_n_c_wis_lengths,
wei_g_k_c_xs_lengths,
out_g_n_k_wos_lengths,
conv_filter_strides,
```
