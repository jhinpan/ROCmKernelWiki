# Diff summary

- **files changed:** 43 (diff was byte-capped; summary is partial)
- **lines:** +648 / -2430
- **kernel-ish files:** 35

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_gemm_xdl_c_shuffle_bias_activation_add.hpp`  (+0/-580)
- `include/ck/tensor_operation/gpu/device/device_gemm_xdl_c_shuffle_bias_activation.hpp`  (+0/-520)
- `include/ck/tensor_operation/gpu/device/device_gemm_xdl_c_shuffle_bias_2d.hpp`  (+0/-513)
- `example/02_gemm_bilinear/gemm_bilinear_xdl_fp16.cpp`  (+305/-0)
- `example/02_gemm_alpha_beta/gemm_xdl_alpha_beta.cpp`  (+0/-252)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_bilinear.hpp`  (+137/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bias2d/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_km_kn_mn_instance.cpp`  (+0/-57)
- `library/src/tensor_operation_instance/gpu/gemm_bias2d/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_km_nk_mn_instance.cpp`  (+0/-57)
- `library/src/tensor_operation_instance/gpu/gemm_bias2d/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_mk_kn_mn_instance.cpp`  (+0/-57)
- `include/ck/tensor_operation/gpu/device/device_gemm_bias_activation_add.hpp`  (+0/-50)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp`  (+26/-24)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp`  (+26/-24)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp`  (+26/-24)
- `include/ck/tensor_operation/gpu/device/device_gemm_bias.hpp`  (+0/-45)
- `include/ck/tensor_operation/gpu/device/device_gemm_bias_activation.hpp`  (+0/-45)

## Key added lines (kernel files)

**`example/02_gemm_bilinear/gemm_bilinear_xdl_fp16.cpp`**
```
struct AlphaBetaAdd
AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta){};
template <typename E, typename C, typename D>
__host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;
```

**`example/03_gemm_bias_relu/gemm_bias_relu_xdl_fp16.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
```

**`include/ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp`**
```
inline std::string getConvForwardSpecializationString(const ConvolutionForwardSpecialization& s)
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm.hpp`**
```
virtual std::unique_ptr<BaseArgument>
MakeArgumentPointer(const void* p_a,
const void* p_b,
void* p_c,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp`**
```
index_t Batch,
CElementwiseOperation c_element_op)
index_t Batch,
CElementwiseOperation c_element_op)
```
