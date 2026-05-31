# Diff summary

- **files changed:** 18
- **lines:** +4771 / -15
- **kernel-ish files:** 17

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_batched_mha_infer_xdl_cshuffle.hpp`  (+1261/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_infer_xdl_cshuffle.hpp`  (+985/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_infer_xdl_cshuffle.hpp`  (+953/-0)
- `example/52_flash_atten_bias/run_grouped_multihead_attention_bias_infer.inc`  (+349/-0)
- `example/52_flash_atten_bias/run_batched_multihead_attention_bias_infer.inc`  (+300/-0)
- `example/52_flash_atten_bias/run_batched_multihead_attention_infer.inc`  (+278/-0)
- `example/52_flash_atten_bias/batched_gemm_multihead_attention_bias_infer.cpp`  (+162/-0)
- `example/52_flash_atten_bias/batched_gemm_multihead_attention_infer.cpp`  (+162/-0)
- `example/52_flash_atten_bias/grouped_mutihead_attention_bias_infer.cpp`  (+161/-0)
- `include/ck/tensor_operation/gpu/device/device_grouped_mha_infer.hpp`  (+75/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_mha_infer.hpp`  (+67/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_fwd_xdl_cshuffle_v2.hpp`  (+8/-9)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_fwd_xdl_cshuffle_v2.hpp`  (+4/-4)
- `example/52_flash_atten_bias/CMakeLists.txt`  (+4/-0)
- `example/52_flash_atten_bias/batched_multihead_attention_bias_forward_v2.cpp`  (+1/-1)

## Key added lines (kernel files)

**`example/52_flash_atten_bias/batched_gemm_multihead_attention_bias_infer.cpp`**
```
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
|-----------------|
|-------------------------------------|
template <ck::index_t... Is>
```

**`example/52_flash_atten_bias/batched_gemm_multihead_attention_infer.cpp`**
```
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
|-----------------|
|-------------------------------------|
template <ck::index_t... Is>
```

**`example/52_flash_atten_bias/grouped_mutihead_attention_bias_infer.cpp`**
```
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
|-----------------|
|-------------------------------------|
template <ck::index_t... Is>
```

**`example/52_flash_atten_bias/run_batched_multihead_attention_bias_infer.inc`**
```
int run(int argc, char* argv[])
bool do_verification = true;
int init_method      = 1;
bool time_kernel     = false;
```

**`example/52_flash_atten_bias/run_batched_multihead_attention_infer.inc`**
```
int run(int argc, char* argv[])
bool do_verification = true;
int init_method      = 1;
bool time_kernel     = false;
```
