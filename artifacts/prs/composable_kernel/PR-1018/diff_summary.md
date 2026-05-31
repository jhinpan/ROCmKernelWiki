# Diff summary

- **files changed:** 16 (diff was byte-capped; summary is partial)
- **lines:** +854 / -266
- **kernel-ish files:** 15

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm/CMakeLists.txt`  (+109/-102)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`  (+91/-46)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instance.cpp`  (+88/-40)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f8_f16_mk_kn_mn_instance.cpp`  (+121/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_fp8_f16_mk_kn_mn_instance.cpp`  (+86/-27)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f8_f16_mk_nk_mn_instance.cpp`  (+110/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instance.cpp`  (+54/-29)
- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+51/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm.hpp`  (+44/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`  (+23/-9)
- `include/ck/utility/type_convert.hpp`  (+30/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`  (+25/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`  (+10/-10)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp`  (+5/-2)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp`  (+5/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp`**
```
<< getGemmSpecializationString(GemmSpec) << ", "
<< PipelineVersionToString[PipelineVer];
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`**
```
PipelineVersion PipelineVer = PipelineVersion::v1,
LoopScheduler LoopSched     = make_default_loop_scheduler()>
index_t K0Padded_,
K0Padded_,
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`**
```
const index_t m_padded  = GridwiseGemm::CalculateMPadded(M);
const index_t n_padded  = GridwiseGemm::CalculateNPadded(N);
const index_t k_padded  = GridwiseGemm::CalculateKPadded(K, K_BATCH);
const index_t k0_padded = GridwiseGemm::CalculateK0Padded(K, K_BATCH);
```

**`include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`**
```
struct PassThroughPack2
template <typename Y, typename X>
__host__ __device__ void operator()(Y& y, const X& x) const;
__host__ __device__ constexpr void operator()(ck::f8x2_t& y, const ck::half2_t& x) const
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`**
```
index_t K0Padded;
index_t K0Padded_,
K0Padded(K0Padded_),
<< "K0Padded:" << K0Padded << ", "
```
