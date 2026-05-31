# Diff summary

- **files changed:** 49
- **lines:** +1100 / -164
- **kernel-ish files:** 41

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_wmma.hpp`  (+499/-0)
- `include/ck/tensor_operation/gpu/warp/wmma_gemm.hpp`  (+146/-1)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp`  (+108/-1)
- `include/ck/utility/amd_wmma.hpp`  (+82/-0)
- `example/01_gemm/gemm_wmma_fp16.cpp`  (+27/-27)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_wmma_cshuffle.hpp`  (+26/-18)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma.hpp`  (+27/-15)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_wmma_cshuffle.hpp`  (+16/-7)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp`  (+15/-8)
- `include/ck/tensor_operation/gpu/grid/gridwise_fpAintB_gemm_wmma.hpp`  (+15/-7)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_softmax_gemm_wmma_cshuffle.hpp`  (+14/-6)
- `include/ck/utility/synchronization.hpp`  (+17/-0)
- `test/wmma_op/wmma_op_util.hpp`  (+16/-0)
- `include/ck/ck.hpp`  (+5/-9)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_wmma_cshuffle.hpp`  (+5/-4)

## Key added lines (kernel files)

**`example/01_gemm/gemm_wmma_fp16.cpp`**
```
< ALayout,
ADataType,
CDataType,
AccDataType,
```

**`example/01_gemm/run_gemm_example.inc`**
```
ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k);
```

**`example/32_batched_gemm_scale_softmax_gemm/cross_attention_forward_wmma_fp16.cpp`**
```
MaskingSpec>
,ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Wmma_CShuffle<
```

**`example/32_batched_gemm_scale_softmax_gemm/self_attention_forward_wmma_fp16.cpp`**
```
MaskingSpec>
,ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Wmma_CShuffle<
```

**`include/ck/host_utility/device_prop.hpp`**
```
inline bool is_gfx12_supported()
return ck::get_device_name() == "gfx1200" || ck::get_device_name() == "gfx1201";
```
