# Diff summary

- **files changed:** 58
- **lines:** +136 / -123
- **kernel-ish files:** 49

## Files (by churn)

- `host/online_compile/CMakeLists.txt`  (+13/-13)
- `host/online_compile/hip_utility/handlehip.cpp`  (+9/-9)
- `host/online_compile/hip_utility/binary_cache.cpp`  (+7/-7)
- `host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.hpp`  (+5/-5)
- `host/online_compile/hip_utility/hip_build_utils.cpp`  (+5/-5)
- `host/online_compile/include/env.hpp`  (+5/-5)
- `host/online_compile/hip_utility/exec_utils.cpp`  (+4/-4)
- `host/online_compile/hip_utility/hipoc_program.cpp`  (+4/-4)
- `host/online_compile/hip_utility/tmp_dir.cpp`  (+4/-4)
- `host/driver_online/CMakeLists.txt`  (+4/-3)
- `host/driver_online/include/online_driver_common.hpp`  (+7/-0)
- `host/online_compile/hip_utility/logger.cpp`  (+3/-3)
- `host/online_compile/hip_utility/target_properties.cpp`  (+3/-3)
- `host/online_compile/include/hip_build_utils.hpp`  (+3/-3)
- `host/online_compile/include/manage_ptr.hpp`  (+3/-3)

## Key added lines (kernel files)

**`host/driver_online/conv_fwd_driver_online.cpp`**
```
online_compile::Handle* handle;
handle = new online_compile::Handle(stream);
```

**`host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v4r4_dlops_nchw_kcyx_nkhw.hpp`**
```
online_compile::Handle* handle,
```

**`host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw.hpp`**
```
online_compile::Handle* handle,
```

**`host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nhwc_kyxc_nhwk.hpp`**
```
online_compile::Handle* handle,
```

**`host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.hpp`**
```
online_compile::Handle* handle,
std::string compile_param_string = get_ck_hip_online_compile_common_flag() + compile_param.GetCompileParameterString();
for(index_t i = 0; i < nrepeat + 1; ++i)
```
