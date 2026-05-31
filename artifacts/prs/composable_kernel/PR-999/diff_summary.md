# Diff summary

- **files changed:** 23
- **lines:** +2783 / -2
- **kernel-ish files:** 20

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle_lds_direct_load.hpp`  (+991/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_lds_direct_load.hpp`  (+414/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_lds_direct_load.hpp`  (+392/-0)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_direct_load.hpp`  (+314/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v4_direct_load.hpp`  (+101/-0)
- `example/01_gemm/gemm_xdl_lds_direct_load_fp16.cpp`  (+58/-0)
- `example/01_gemm/gemm_xdl_lds_direct_load_fp32.cpp`  (+57/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_lds_direct_load_f16_f16_f16_mk_nk_mn_instance.cpp`  (+54/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_mk_nk_mn_instance.cpp`  (+53/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_km_kn_mn_instance.cpp`  (+51/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_km_nk_mn_instance.cpp`  (+51/-0)
- `example/04_gemm_add_add_fastgelu/gemm_add_add_fastgelu_xdl_lds_direct_load_fp32.cpp`  (+50/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_mk_kn_mn_instance.cpp`  (+50/-0)
- `include/ck/utility/amd_buffer_addressing.hpp`  (+37/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`  (+34/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_lds_direct_load_fp16.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using ADataType        = F16;
using BDataType        = F16;
```

**`example/01_gemm/gemm_xdl_lds_direct_load_fp32.cpp`**
```
using F32 = float;
using ADataType        = F32;
using BDataType        = F32;
using AccDataType      = F32;
```

**`example/04_gemm_add_add_fastgelu/gemm_add_add_fastgelu_xdl_lds_direct_load_fp32.cpp`**
```
using ADataType        = F32;
using BDataType        = F32;
using AccDataType      = F32;
using CShuffleDataType = F32;
```

**`example/04_gemm_add_add_fastgelu/run_gemm_add_add_fastgelu_example.inc`**
```
std::cerr << device_op.GetTypeString() << " does not support this problem" << std::endl;
return true;
```

**`include/ck/host_utility/device_prop.hpp`**
```
inline bool is_lds_direct_load_supported()
return ck::get_device_name() == "gfx90a" || ck::get_device_name() == "gfx940" ||
ck::get_device_name() == "gfx941" || ck::get_device_name() == "gfx942";
```
