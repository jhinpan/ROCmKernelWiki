# Diff summary

- **files changed:** 36
- **lines:** +1133 / -45
- **kernel-ish files:** 33

## Files (by churn)

- `example/09_convnd_fwd/convnd_fwd_xdl_fp64.cpp`  (+344/-0)
- `example/01_gemm/gemm_xdl_fp64.cpp`  (+240/-0)
- `test/gemm/gemm_xdl_fp64.cpp`  (+156/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f64_f64_f64_mk_nk_mn_instance.cpp`  (+54/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f64_f64_f64_km_kn_mn_instance.cpp`  (+49/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f64_f64_f64_km_nk_mn_instance.cpp`  (+49/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f64_f64_f64_mk_kn_mn_instance.cpp`  (+49/-0)
- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+33/-3)
- `include/ck/utility/amd_xdlops.hpp`  (+19/-0)
- `profiler/src/profile_gemm.cpp`  (+16/-0)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_gemm.hpp`  (+7/-6)
- `profiler/include/profile_gemm_impl.hpp`  (+10/-2)
- `test/gemm/gemm_dl_fp16.cpp`  (+8/-3)
- `test/gemm/gemm_dl_fp32.cpp`  (+8/-3)
- `test/gemm/gemm_dl_int8.cpp`  (+8/-3)

## Key added lines (kernel files)

**`example/01_gemm/gemm_dl_fp16.cpp`**
```
ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;
```

**`example/01_gemm/gemm_dl_fp32.cpp`**
```
ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;
```

**`example/01_gemm/gemm_dl_int8.cpp`**
```
ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;
```

**`example/01_gemm/gemm_xdl_bf16.cpp`**
```
ReferenceGemm<float, float, float, float, PassThrough, PassThrough, PassThrough>;
```

**`example/01_gemm/gemm_xdl_fp16.cpp`**
```
ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;
```
