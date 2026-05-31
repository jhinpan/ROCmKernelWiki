# Diff summary

- **files changed:** 12
- **lines:** +2771 / -86
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_data_to_gemm.hpp`  (+1064/-0)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp`  (+985/-0)
- `include/ck_tile/host/reference/reference_grouped_conv_bwd_data.hpp`  (+227/-0)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data.cpp`  (+216/-0)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_bwd_data_example.inc`  (+188/-0)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+41/-44)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+42/-42)
- `example/ck_tile/20_grouped_convolution/CMakeLists.txt`  (+3/-0)
- `include/ck_tile/ops/grouped_convolution.hpp`  (+2/-0)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+1/-0)
- `include/ck_tile/host.hpp`  (+1/-0)
- `include/ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data.cpp`**
```
template <ck_tile::index_t NDimSpatial,
typename InDataType,
typename WeiDataType,
typename AccDataType,
```

**`example/ck_tile/20_grouped_convolution/run_grouped_convolution_bwd_data_example.inc`**
```
template <ck_tile::index_t NDimSpatial,
typename InDataType,
typename WeiDataType,
typename AccDataType,
```

**`include/ck_tile/core/tensor/tensor_view.hpp`**
```
memory_operation_enum DstInMemOp      = memory_operation_enum::set,
```

**`include/ck_tile/host/reference/reference_grouped_conv_bwd_data.hpp`**
```
namespace ck_tile {
template <ck_tile::index_t NDimSpatial,
typename InDataType,
typename WeiDataType,
```

**`include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp`**
```
namespace ck_tile {
template <typename GroupedConvTraitsType_, typename TilePartitioner_>
struct GroupedConvBwdDataKernelArgs
using TilePartitioner = remove_cvref_t<TilePartitioner_>;
```
