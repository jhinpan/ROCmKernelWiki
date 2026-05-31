# Diff summary

- **files changed:** 13
- **lines:** +148 / -42
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp`  (+18/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_b_preshuffle.hpp`  (+18/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_b_scale.hpp`  (+18/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp`  (+18/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl_fpAintB_b_scale.hpp`  (+12/-4)
- `example/01_gemm/gemm_xdl_fp8_pk_i4_bpreshuffle_v3.cpp`  (+10/-3)
- `example/01_gemm/gemm_xdl_bf16_pk_i4_v3.cpp`  (+9/-3)
- `example/01_gemm/gemm_xdl_fp16_pk_i4_v3.cpp`  (+9/-3)
- `example/01_gemm/gemm_xdl_fp16_pk_i4_v3_b_scale.cpp`  (+9/-3)
- `example/01_gemm/gemm_xdl_fp8_pk_i4_v3.cpp`  (+9/-3)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_pk_i4.cpp`  (+8/-3)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_pk_i4.cpp`  (+8/-3)
- `example/24_batched_gemm/run_batched_gemm_example_fp16int4_b_scale.inc`  (+2/-1)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_bf16_pk_i4_v3.cpp`**
```
DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n_permute.mDesc.GetElementSpaceSize() / 2);
if(!gemm.IsSupportedArgument(argument))
if(!(ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950"))
std::cout << "This kernel support gfx942 and gfx950 only" << std::endl;
```

**`example/01_gemm/gemm_xdl_fp16_pk_i4_v3.cpp`**
```
DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n_permute.mDesc.GetElementSpaceSize() / 2);
if(!gemm.IsSupportedArgument(argument))
if(!(ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950"))
std::cout << "This kernel support gfx942 and gfx950 only" << std::endl;
```

**`example/01_gemm/gemm_xdl_fp16_pk_i4_v3_b_scale.cpp`**
```
DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n_permute.mDesc.GetElementSpaceSize() / 2);
if(!gemm.IsSupportedArgument(argument))
if(!(ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950"))
std::cout << "This kernel support gfx942 and gfx950 only" << std::endl;
```

**`example/01_gemm/gemm_xdl_fp8_pk_i4_bpreshuffle_v3.cpp`**
```
DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n_preshuffled.mDesc.GetElementSpaceSize() /
if(!gemm.IsSupportedArgument(argument))
if(!(ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950"))
std::cout << "This kernel support gfx942 and gfx950 only" << std::endl;
```

**`example/01_gemm/gemm_xdl_fp8_pk_i4_v3.cpp`**
```
DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n_permute.mDesc.GetElementSpaceSize() / 2);
if(!gemm.IsSupportedArgument(argument))
if(!(ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950"))
std::cout << "This kernel support gfx942 and gfx950 only" << std::endl;
```
