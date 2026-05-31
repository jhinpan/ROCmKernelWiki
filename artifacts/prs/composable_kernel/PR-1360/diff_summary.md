# Diff summary

- **files changed:** 28 (diff was byte-capped; summary is partial)
- **lines:** +4320 / -2
- **kernel-ish files:** 24

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_streamk_v3.hpp`  (+2010/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_streamk_v3.hpp`  (+556/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_streamk.hpp`  (+337/-0)
- `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`  (+322/-0)
- `example/01_gemm/run_gemm_example_streamk_v2.inc`  (+298/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f16_f16/device_gemm_xdl_universal_streamk_f16_f16_f16_mk_nk_mn.hpp`  (+98/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f16_f16/device_gemm_xdl_universal_streamk_f16_f16_f16_mk_kn_mn.hpp`  (+91/-0)
- `example/01_gemm/common.hpp`  (+67/-2)
- `example/01_gemm/gemm_xdl_fp16_streamk_v3.cpp`  (+48/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_streamk_v2.hpp`  (+44/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f16_f16/device_gemm_xdl_universal_streamk_f16_f16_f16_mk_kn_mn_mem_v1_default_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f16_f16/device_gemm_xdl_universal_streamk_f16_f16_f16_mk_kn_mn_mem_v1_kpadding_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f16_f16/device_gemm_xdl_universal_streamk_f16_f16_f16_mk_kn_mn_mem_v1_mnkpadding_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f16_f16/device_gemm_xdl_universal_streamk_f16_f16_f16_mk_kn_mn_mem_v2_default_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f16_f16/device_gemm_xdl_universal_streamk_f16_f16_f16_mk_kn_mn_mem_v2_kpadding_instance.cpp`  (+31/-0)

## Key added lines (kernel files)

**`example/01_gemm/common.hpp`**
```
struct ProblemSizeStreamK_universal final
ck::index_t M = 3840;
ck::index_t N = 4096;
ck::index_t K = 4096;
```

**`example/01_gemm/gemm_xdl_fp16_streamk_v3.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```

**`example/01_gemm/run_gemm_example_streamk_v2.inc`**
```
template <typename DataType>
inline __host__ __device__ constexpr double get_rtol()
if constexpr(std::is_same_v<DataType, float>)
return 1e-3;
```

**`include/ck/tensor_operation/gpu/device/device_gemm_streamk_v2.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_streamk_v3.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```
