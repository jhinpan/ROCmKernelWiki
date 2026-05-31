# Diff summary

- **files changed:** 14
- **lines:** +2176 / -65
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+861/-0)
- `include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_weight_to_gemm.hpp`  (+659/-0)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight.cpp`  (+218/-0)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_bwd_weight_example.inc`  (+188/-0)
- `include/ck_tile/host/reference/reference_grouped_conv_bwd_weight.hpp`  (+167/-0)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+19/-20)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_fwd_example.inc`  (+9/-29)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_utils.hpp`  (+25/-2)
- `include/ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp`  (+11/-6)
- `include/ck_tile/ops/gemm_group_quant.hpp`  (+7/-3)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward.cpp`  (+4/-4)
- `example/ck_tile/20_grouped_convolution/CMakeLists.txt`  (+5/-1)
- `include/ck_tile/ops/grouped_convolution.hpp`  (+2/-0)
- `include/ck_tile/host.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight.cpp`**
```
template <ck_tile::index_t NDimSpatial,
typename InDataType,
typename WeiDataType,
typename AccDataType,
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_forward.cpp`**
```
float grouped_conv_fwd(const ck_tile::GroupedConvFwdHostArgs& args, const ck_tile::stream_config& s)
const dim3 grids      = Kernel::GridSize(kargs);
std::string wei_layout = arg_parser.get_str("wei_layout");
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_utils.hpp`**
```
template <typename InDataType, typename WeiDataType, typename AccDataType, typename OutDataType>
auto calculate_rtol_atol(const ck_tile::index_t GemmK,
const ck_tile::index_t kbatch,
const float max_accumulated_value)
```

**`example/ck_tile/20_grouped_convolution/run_grouped_convolution_bwd_weight_example.inc`**
```
template <ck_tile::index_t NDimSpatial,
typename InDataType,
typename WeiDataType,
typename AccDataType,
```

**`example/ck_tile/20_grouped_convolution/run_grouped_convolution_fwd_example.inc`**
```
float invoke_grouped_conv_fwd(const ck_tile::GroupedConvFwdHostArgs& args,
int n_warmup,
int n_repeat)
ck_tile::GroupedConvFwdHostArgs args(conv_param,
```
