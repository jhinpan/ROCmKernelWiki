# Diff summary

- **files changed:** 41
- **lines:** +3358 / -517
- **kernel-ish files:** 33

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d_xdl_cshuffle.hpp`  (+750/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp`  (+668/-0)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7.hpp`  (+295/-0)
- `profiler/include/profile_gemm_add_add_fastgelu_impl.hpp`  (+288/-0)
- `example/03_gemm_bias_relu/gemm_xdl_bias_relu.cpp`  (+154/-115)
- `example/04_gemm_bias_relu_add/gemm_xdl_bias_relu_add.cpp`  (+0/-257)
- `example/04_gemm_add_add_fastgelu/gemm_add_add_fastgelu_xdl_fp16.cpp`  (+245/-0)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp`  (+169/-0)
- `profiler/src/profile_gemm_add_add_fastgelu.cpp`  (+152/-0)
- `include/ck/tensor_operation/gpu/element/element_wise_operation.hpp`  (+83/-26)
- `include/ck/utility/tuple.hpp`  (+49/-21)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp`  (+66/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp`  (+66/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp`  (+66/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp`  (+63/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_fp16.cpp`**
```
using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
```

**`example/03_gemm_bias_relu/gemm_xdl_bias_relu.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`example/04_gemm_add_add_fastgelu/gemm_add_add_fastgelu_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`include/ck/tensor_description/tensor_adaptor.hpp`**
```
__host__ __device__ constexpr TensorAdaptor() : transforms_{}, element_size_{} {}
```

**`include/ck/tensor_description/tensor_descriptor.hpp`**
```
__host__ __device__ constexpr TensorDescriptor()
: transforms_{}, element_size_{}, element_space_size_{}
```
