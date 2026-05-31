# Diff summary

- **files changed:** 29 (diff was byte-capped; summary is partial)
- **lines:** +2038 / -230
- **kernel-ish files:** 27

## Files (by churn)

- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r5.hpp`  (+655/-0)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer_v1r4.hpp`  (+522/-0)
- `example/2_gemm_xdl_bias_relu_add/gemm_xdl_bias_relu_add.cpp`  (+247/-0)
- `example/1_gemm_xdl/gemm_xdl.cpp`  (+72/-20)
- `device_operation/include/device_gemm_xdl.hpp`  (+57/-7)
- `example/2_gemm_xdl_bias_relu_add/README.md`  (+61/-0)
- `device_operation/include/device_conv_fwd_xdl_nhwc_kyxc_nhwk.hpp`  (+51/-8)
- `device_operation/include/device_conv.hpp`  (+38/-6)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`  (+39/-4)
- `device_operation/device_conv_xdl_instance_f16_f16_f16_nhwc_kyxc_nhwk.cpp`  (+21/-18)
- `device_operation/device_conv_xdl_instance_f32_f32_f32_nhwc_kyxc_nhwk.cpp`  (+21/-18)
- `device_operation/device_gemm_xdl_instance_f16_f16_f16_mk_nk_mn.cpp`  (+21/-18)
- `device_operation/device_gemm_xdl_instance_f32_f32_f32_mk_nk_mn.cpp`  (+21/-18)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer_v3r2.hpp`  (+24/-9)
- `device_operation/include/device_gemm.hpp`  (+21/-10)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer.hpp`**
```
typename SrcElementwiseOperation,
__device__ constexpr BlockwiseTensorSliceTransfer_v4(
const SrcDesc& src_desc,
const Index& src_block_slice_origin,
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`**
```
typename AElementwiseOperation,
typename BElementwiseOperation,
typename CElementwiseOperation,
const AElementwiseOperation a_element_op,
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r5.hpp`**
```
namespace ck {
template <typename GridwiseGemm,
typename FloatAB,
typename FloatC,
```

**`composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer.hpp`**
```
typename DstElementwiseOperation,
__device__ constexpr ThreadwiseTensorSliceTransfer_v1r3(
const DstDesc& dst_desc,
const Index& dst_slice_origin_idx,
```

**`composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer_v1r4.hpp`**
```
namespace ck {
template <typename SrcData,
typename DstData,
typename SrcDesc,
```
