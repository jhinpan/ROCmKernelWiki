# Diff summary

- **files changed:** 14
- **lines:** +1689 / -211
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+1008/-0)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`  (+397/-0)
- `include/ck/tensor_operation/gpu/device/matrix_padder.hpp`  (+156/-137)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp`  (+59/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm.hpp`  (+0/-28)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp`  (+0/-27)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_xdl_fp16.cpp`  (+12/-11)
- `include/ck/tensor_operation/gpu/device/device_batched_contraction_multiple_d_xdl_cshuffle.hpp`  (+19/-0)
- `include/ck/tensor_operation/gpu/device/gemm_specialization.hpp`  (+18/-0)
- `include/ck/utility/functional.hpp`  (+14/-0)
- `example/31_batched_gemm_gemm/batched_gemm_gemm_xdl_fp16.cpp`  (+5/-4)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp`  (+0/-2)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_softmax_gemm_xdl_cshuffle_v1.hpp`  (+0/-2)
- `example/32_batched_gemm_scale_softmax_gemm/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/31_batched_gemm_gemm/batched_gemm_gemm_xdl_fp16.cpp`**
```
DeviceMem a_g_m_k_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize());
DeviceMem b0_g_k_n_device_buf(sizeof(B0DataType) * b0_g_k_n.mDesc.GetElementSpaceSize());
DeviceMem b1_g_n_o_device_buf(sizeof(B1DataType) * b1_g_n_o.mDesc.GetElementSpaceSize());
DeviceMem c_g_m_o_device_buf(sizeof(CDataType) *
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`**
```
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
|-----------------|
|-------------------------------------|
template <ck::index_t... Is>
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_xdl_fp16.cpp`**
```
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
|-----------------|
|-------------------------------------|
printf("arg4 to 16: M, N, K, O, Batch, StrideA, StrideB0, StrideB1, StrideC, BatchStrideA, "
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```
