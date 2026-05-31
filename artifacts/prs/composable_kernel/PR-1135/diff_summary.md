# Diff summary

- **files changed:** 17
- **lines:** +3015 / -17
- **kernel-ish files:** 16

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v2.hpp`  (+1153/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops.hpp`  (+999/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v2.hpp`  (+306/-0)
- `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`  (+301/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instance.cpp`  (+82/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instance.cpp`  (+73/-8)
- `example/01_gemm/gemm_xdl_fp16_v2.cpp`  (+51/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`  (+30/-0)
- `include/ck/stream_config.hpp`  (+2/-2)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instance.cpp`  (+3/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instance.cpp`  (+3/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp`  (+3/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp`  (+3/-1)
- `example/01_gemm/CMakeLists.txt`  (+3/-0)
- `example/35_splitK_gemm/run_splitK_gemm_example.inc`  (+1/-1)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_fp16_v2.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```

**`example/35_splitK_gemm/run_splitK_gemm_example.inc`**
```
float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel, 1});
```

**`example/35_splitK_gemm/splitK_gemm_xdl_fp16.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::KPadding;
```

**`include/ck/stream_config.hpp`**
```
int cold_niters_       = 5;
int nrepeat_           = 50;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops.hpp`**
```
namespace ck {
template <index_t BlockSize,
index_t MPerBlock,
index_t NPerBlock,
```
