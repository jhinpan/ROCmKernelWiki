# Diff summary

- **files changed:** 17
- **lines:** +1133 / -269
- **kernel-ish files:** 13

## Files (by churn)

- `example/32_batched_gemm_scale_softmax_gemm/run_batched_gemm_scale_softmax_gemm.inc`  (+261/-0)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_xdl_fp16.cpp`  (+2/-257)
- `test/batched_gemm_softmax_gemm_permute/test_batched_gemm_softmax_gemm_permute_bf16.cpp`  (+182/-0)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_permute_xdl_bf16.cpp`  (+159/-0)
- `test/batched_gemm_softmax_gemm_permute/test_batched_gemm_softmax_gemm_permute_util.hpp`  (+144/-2)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_xdl_bf16.cpp`  (+143/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle_bf16_bf16_bf16_bf16_gmk_gnk_gno_gmo_instance.cpp`  (+133/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute.hpp`  (+56/-0)
- `profiler/include/profile_batched_gemm_softmax_gemm_permute_impl.hpp`  (+19/-2)
- `example/32_batched_gemm_scale_softmax_gemm/run_batched_gemm_scale_softmax_gemm_permute.inc`  (+17/-1)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_softmax.hpp`  (+7/-5)
- `test/batched_gemm_softmax_gemm_permute/CMakeLists.txt`  (+4/-1)
- `example/32_batched_gemm_scale_softmax_gemm/CMakeLists.txt`  (+4/-0)
- `include/ck/utility/amd_xdlops.hpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_permute_xdl_bf16.cpp`**
```
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
|-----------------|
|-------------------------------------|
template <ck::index_t... Is>
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_xdl_bf16.cpp`**
```
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
|-----------------|
|-------------------------------------|
template <ck::index_t... Is>
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_xdl_fp16.cpp`**
```
int main(int argc, char* argv[]) { return run(argc, argv); }
```

**`example/32_batched_gemm_scale_softmax_gemm/run_batched_gemm_scale_softmax_gemm.inc`**
```
int run(int argc, char* argv[])
bool do_verification = true;
int init_method      = 2;
bool time_kernel     = false;
```

**`example/32_batched_gemm_scale_softmax_gemm/run_batched_gemm_scale_softmax_gemm_permute.inc`**
```
double rtol = 1e-3;
double atol = 1e-3;
if(std::is_same_v<ADataType, ck::bhalf_t> && std::is_same_v<B0DataType, ck::bhalf_t> &&
std::is_same_v<B1DataType, ck::bhalf_t> && std::is_same_v<CDataType, ck::bhalf_t>)
```
