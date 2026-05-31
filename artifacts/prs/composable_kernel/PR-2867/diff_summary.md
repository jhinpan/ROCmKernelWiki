# Diff summary

- **files changed:** 30 (diff was byte-capped; summary is partial)
- **lines:** +1052 / -440
- **kernel-ish files:** 30

## Files (by churn)

- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bias_clamp_xdl.inc`  (+176/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bias_clamp.hpp`  (+71/-39)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp.hpp`  (+71/-38)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_b_scale.hpp`  (+51/-47)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp`  (+50/-46)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp_xdl.inc`  (+75/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_ab_scale.hpp`  (+32/-28)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1.hpp`  (+30/-26)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp`  (+33/-20)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v4_b_scale.hpp`  (+25/-23)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_ab_scale.hpp`  (+24/-22)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v4.hpp`  (+24/-22)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v5.hpp`  (+24/-22)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp`  (+26/-19)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_abd_xdl_cshuffle.hpp`  (+22/-18)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp`**
```
using ComputeDataTypeBuf =
conditional_t<std::is_same<ComputeDataType, ck::tf32_t>::value, float, ComputeDataType>;
ComputeDataTypeBuf,
ComputeDataTypeBuf,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1.hpp`**
```
using ComputeDataTypeBuf = typename Base::ComputeDataTypeBuf;
auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_ab_scale.hpp`**
```
using ComputeDataTypeBuf = typename Base::ComputeDataTypeBuf;
auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_b_scale.hpp`**
```
using ComputeDataTypeBuf = typename Base::ComputeDataTypeBuf;
auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp`**
```
using ComputeDataTypeBuf = typename Base::ComputeDataTypeBuf;
auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
```
