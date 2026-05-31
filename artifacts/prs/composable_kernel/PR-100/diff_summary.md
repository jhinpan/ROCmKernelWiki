# Diff summary

- **files changed:** 28
- **lines:** +1426 / -149
- **kernel-ish files:** 24

## Files (by churn)

- `example/1_gemm_xdl/gemm_xdl_bf16.cpp`  (+235/-0)
- `example/1_gemm_xdl/gemm_xdl_int8.cpp`  (+226/-0)
- `test/gemm_xdl/test_gemm_bf16.cpp`  (+163/-0)
- `test/gemm_xdl/test_gemm_fp32.cpp`  (+138/-0)
- `test/gemm_xdl/test_gemm_int8.cpp`  (+137/-0)
- `profiler/include/profile_gemm_impl.hpp`  (+101/-17)
- `device_operation/src/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instance.cpp`  (+53/-52)
- `test/gemm_xdl/gemm_util.hpp`  (+103/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instance.cpp`  (+56/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_int8_int8_int8_mk_nk_mn_instance.cpp`  (+55/-0)
- `profiler/src/profile_gemm.cpp`  (+45/-3)
- `composable_kernel/include/utility/amd_buffer_addressing.hpp`  (+22/-22)
- `composable_kernel/include/utility/data_type.hpp`  (+12/-11)
- `host/host_tensor/include/host_tensor_generator.hpp`  (+9/-10)
- `test/CMakeLists.txt`  (+18/-0)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/element_wise_operation.hpp`**
```
__host__ __device__ void operator()(bhalf_t& y, const bhalf_t& x) const { y = x; }
```

**`composable_kernel/include/tensor_operation/xdlops_gemm.hpp`**
```
static constexpr auto GetMfma<bhalf_t, 32, 32>()
static constexpr auto GetMfma<bhalf_t, 16, 16>()
is_same<base_type, bhalf_t>::value || is_same<base_type, int8_t>::value,
"base base_type must be float, half, bfloat16, and int8_t!");
```

**`composable_kernel/include/utility/amd_buffer_addressing.hpp`**
```
__device__ bhalf_t
__device__ bhalf2_t
__device__ bhalf4_t
llvm_amdgcn_raw_buffer_store_i16(bhalf_t vdata,
```

**`composable_kernel/include/utility/amd_xdlops.hpp`**
```
__device__ static void Run(const bhalf4_t& reg_a, const bhalf4_t& reg_b, FloatC& reg_c)
__device__ static void Run(const bhalf4_t& reg_a, const bhalf4_t& reg_b, FloatC& reg_c)
__device__ static void Run(const bhalf2_t& reg_a, const bhalf2_t& reg_b, FloatC& reg_c)
__device__ static void Run(const bhalf2_t& reg_a, const bhalf2_t& reg_b, FloatC& reg_c)
```

**`composable_kernel/include/utility/data_type.hpp`**
```
using bhalf_t = ushort;
using half_t  = _Float16;
struct scalar_type<bhalf_t>
using type                           = bhalf_t;
```
