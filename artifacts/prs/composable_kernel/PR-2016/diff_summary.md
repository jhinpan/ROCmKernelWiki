# Diff summary

- **files changed:** 8
- **lines:** +140 / -41
- **kernel-ish files:** 6

## Files (by churn)

- `example/67_gemm_microscaling/gemm_mx_fp8_e8m0_scale.cpp`  (+42/-0)
- `example/67_gemm_microscaling/gemm_mx_fp8_fp8_scale.cpp`  (+42/-0)
- `example/67_gemm_microscaling/gemm_mx_common.hpp`  (+21/-10)
- `include/ck/utility/dtype_vector.hpp`  (+8/-22)
- `example/67_gemm_microscaling/README.md`  (+13/-5)
- `example/67_gemm_microscaling/CMakeLists.txt`  (+7/-2)
- `include/ck/utility/data_type.hpp`  (+7/-0)
- `example/67_gemm_microscaling/gemm_mx_fp8_fp16_scale.cpp`  (+0/-2)

## Key added lines (kernel files)

**`example/67_gemm_microscaling/gemm_mx_common.hpp`**
```
int init_method     = 2;     // (0=constant values, 1=integer values, 2=decimal values)
<< "arg2: initialization (0=constant values, 1=integer values, 2=decimal values)"
<< "arg5 to 10: M(128x), N(128x), K(64x), StrideA, StrideB, StrideC" << std::endl
a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 6}); // Z[-5,5]
```

**`example/67_gemm_microscaling/gemm_mx_fp8_e8m0_scale.cpp`**
```
using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
using XDataType = ck::e8m0_bexp_t;
using CDataType        = ck::half_t;
```

**`example/67_gemm_microscaling/gemm_mx_fp8_fp8_scale.cpp`**
```
using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
using XDataType = ck::f8_t;
using CDataType        = ck::half_t;
```

**`include/ck/utility/data_type.hpp`**
```
template <>
struct scalar_type<e8m0_bexp_t>
using type                           = e8m0_bexp_t::type;
static constexpr index_t vector_size = 1;
```

**`include/ck/utility/dtype_vector.hpp`**
```
template <>
struct nnvb_data_t_selector<e8m0_bexp_t>
using type = e8m0_bexp_t::type;
struct scalar_type<non_native_vector_base<T, N>>
```
