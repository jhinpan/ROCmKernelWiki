# Diff summary

- **files changed:** 12
- **lines:** +97 / -117
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops.hpp`  (+18/-23)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp`  (+13/-15)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v4.hpp`  (+11/-14)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v5.hpp`  (+11/-14)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp`  (+10/-13)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_wmma.hpp`  (+7/-9)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1.hpp`  (+7/-8)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp`  (+5/-6)
- `test/grouped_convnd_fwd/test_grouped_convnd_fwd_multi_ab_interface.cpp`  (+5/-5)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops_skip_b_lds.hpp`  (+4/-5)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_dpp.hpp`  (+4/-4)
- `cmake/EnableCompilerWarnings.cmake`  (+2/-1)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_dpp.hpp`**
```
dpp_gemm.Run(a_thread_vec.template AsType<dpp_input_type>(),
b_thread_vec.template AsType<dpp_input_type>(),
c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops.hpp`**
```
xdlops_gemm.Run(
xdlops_gemm.Run(
xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
b_thread_vec.template AsType<mfma_input_type>(),
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1.hpp`**
```
xdlops_gemm.Run(
xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
b_thread_vec.template AsType<mfma_input_type>(),
c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp`**
```
xdlops_gemm.Run(
xdlops_gemm.Run(
xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
b_thread_vec.template AsType<mfma_input_type>(),
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp`**
```
xdlops_gemm.Run(
xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
b_thread_vec.template AsType<mfma_input_type>(),
c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
```
