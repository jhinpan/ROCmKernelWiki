# Diff summary

- **files changed:** 12
- **lines:** +62 / -32
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_convnd_fwd_xdl_nhwc_kyxc_nhwk.hpp`  (+23/-3)
- `include/ck/tensor_operation/gpu/device/device_gemm_xdl.hpp`  (+18/-2)
- `example/01_gemm/gemm_xdl_fp64.cpp`  (+4/-6)
- `example/01_gemm/gemm_xdl_bf16.cpp`  (+3/-3)
- `example/01_gemm/gemm_xdl_fp16.cpp`  (+3/-3)
- `example/01_gemm/gemm_xdl_int8.cpp`  (+3/-3)
- `example/01_gemm/gemm_dl_fp16.cpp`  (+1/-3)
- `example/01_gemm/gemm_dl_fp32.cpp`  (+1/-3)
- `example/01_gemm/gemm_dl_int8.cpp`  (+1/-3)
- `example/01_gemm/CMakeLists.txt`  (+2/-1)
- `example/09_convnd_fwd/CMakeLists.txt`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/device_gemm_dl.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/01_gemm/gemm_dl_fp16.cpp`**
```
std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;
```

**`example/01_gemm/gemm_dl_fp32.cpp`**
```
std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;
```

**`example/01_gemm/gemm_dl_int8.cpp`**
```
std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;
```

**`example/01_gemm/gemm_xdl_bf16.cpp`**
```
std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;
return 0;
```

**`example/01_gemm/gemm_xdl_fp16.cpp`**
```
std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;
return 0;
```
