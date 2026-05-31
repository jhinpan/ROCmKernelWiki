# Diff summary

- **files changed:** 17
- **lines:** +1073 / -12
- **kernel-ish files:** 16

## Files (by churn)

- `profiler/include/profiler/profile_gemm_add_fastgelu_impl.hpp`  (+232/-0)
- `profiler/include/profiler/profile_gemm_fastgelu_impl.hpp`  (+222/-0)
- `profiler/src/profile_gemm_add_fastgelu.cpp`  (+146/-0)
- `profiler/src/profile_gemm_fastgelu.cpp`  (+137/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_kn_mn_mn_mn_instance.cpp`  (+28/-1)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_nk_mn_mn_mn_instance.cpp`  (+28/-1)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instance.cpp`  (+28/-1)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instance.cpp`  (+28/-1)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instance.cpp`  (+28/-1)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn_instance.cpp`  (+28/-1)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn_instance.cpp`  (+28/-1)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn_instance.cpp`  (+28/-1)
- `library/src/tensor_operation_instance/gpu/gemm_fastgelu/device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp`  (+27/-1)
- `library/src/tensor_operation_instance/gpu/gemm_fastgelu/device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp`  (+27/-1)
- `library/src/tensor_operation_instance/gpu/gemm_fastgelu/device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp`  (+27/-1)

## Key added lines (kernel files)

**`library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_kn_mn_mn_mn_instance.cpp`**
```
static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
using device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_kn_mn_mn_mn_irregular_tile_instances =
std::tuple<
```

**`library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_nk_mn_mn_mn_instance.cpp`**
```
static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
using device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_nk_mn_mn_mn_irregular_tile_instances =
std::tuple<
```

**`library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instance.cpp`**
```
static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
using device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_irregular_tile_instances =
std::tuple<
```

**`library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instance.cpp`**
```
static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
using device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_irregular_tile_instances =
std::tuple<
```

**`library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instance.cpp`**
```
static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
using device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_irregular_tile_instances =
std::tuple<
```
