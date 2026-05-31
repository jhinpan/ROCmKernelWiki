# Diff summary

- **files changed:** 11
- **lines:** +524 / -41
- **kernel-ish files:** 10

## Files (by churn)

- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_fwd_bias_clamp_example.inc`  (+301/-0)
- `include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`  (+66/-0)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_bias_clamp.cpp`  (+58/-0)
- `include/ck_tile/host/reference/reference_grouped_conv_fwd.hpp`  (+25/-8)
- `include/ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp`  (+20/-9)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+17/-7)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_fwd_example.inc`  (+7/-7)
- `test/ck_tile/elementwise/test_elementwise_1d.cpp`  (+13/-1)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`  (+7/-6)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+7/-3)
- `example/ck_tile/20_grouped_convolution/CMakeLists.txt`  (+3/-0)

## Key added lines (kernel files)

**`example/ck_tile/20_grouped_convolution/grouped_convolution_forward_bias_clamp.cpp`**
```
template <template <typename PrecType> typename GemmConfig>
int run_grouped_conv_fwd_bias_clamp_example(int argc, char* argv[])
using Invoker = GroupedConvolutionForwardInvoker;
auto [result, arg_parser] = create_args(argc, argv);
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`**
```
typename DsDataType    = ck_tile::tuple<>,
typename DsLayout      = ck_tile::tuple<>,
typename CDElementWise = ck_tile::element_wise::PassThrough>
static float grouped_conv_fwd(const ck_tile::GroupedConvFwdHostArgs<CDElementWise>& args,
```

**`example/ck_tile/20_grouped_convolution/run_grouped_convolution_fwd_bias_clamp_example.inc`**
```
using BiasAndClamp = ck_tile::element_wise::
Compose<ck_tile::element_wise::MultiDAdd, ck_tile::element_wise::Clamp, true>;
template <ck_tile::index_t NDimSpatial,
typename GemmWarpConfig,
```

**`example/ck_tile/20_grouped_convolution/run_grouped_convolution_fwd_example.inc`**
```
float invoke_grouped_conv_fwd(const ck_tile::GroupedConvFwdHostArgs<>& args,
ck_tile::GroupedConvFwdHostArgs<> args(conv_param,
input_dev_buf.GetDeviceBuffer(),
weight_dev_buf.GetDeviceBuffer(),
```

**`include/ck_tile/host/reference/reference_grouped_conv_fwd.hpp`**
```
typename OutDataType,
typename Elfunc = ck_tile::element_wise::PassThrough,
typename Tuple  = ck_tile::tuple<>>
std::vector<ck_tile::long_index_t>,
```
