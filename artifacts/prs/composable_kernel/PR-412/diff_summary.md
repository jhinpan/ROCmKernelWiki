# Diff summary

- **files changed:** 9
- **lines:** +1499 / -415
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+929/-0)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`  (+443/-0)
- `example/32_batched_gemm_scale_softmax_gemm/padded_batched_gemm_scale_softmax_gemm_xdl_fp16.cpp`  (+0/-397)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute.hpp`  (+69/-0)
- `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`  (+44/-0)
- `example/32_batched_gemm_scale_softmax_gemm/CMakeLists.txt`  (+5/-5)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_xdl_fp16.cpp`  (+4/-4)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_gemm_xdl_cshuffle.hpp`  (+2/-6)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`  (+3/-3)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`**
```
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNPadding;
ck::index_t M             = 120;
ck::index_t N             = 1000;
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_xdl_fp16.cpp`**
```
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNPadding;
GemmSpec,
ck::index_t M             = 1020;
ck::index_t N             = 1020;
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`**
```
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
|-----------------|
|-------------------------------------|
template <ck::index_t... Is>
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_gemm_xdl_cshuffle.hpp`**
```
if(!DeviceOp::IsSupportedArgument(arg))
throw std::runtime_error("wrong! unsupported argument");
```

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```
