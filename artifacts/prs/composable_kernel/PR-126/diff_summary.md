# Diff summary

- **files changed:** 20
- **lines:** +1917 / -0
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_xdl.hpp`  (+562/-0)
- `profiler/include/profile_grouped_gemm_impl.hpp`  (+314/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_fp16.cpp`  (+234/-0)
- `test/grouped_gemm/grouped_gemm_fp16.cpp`  (+213/-0)
- `profiler/src/profile_grouped_gemm.cpp`  (+157/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp`  (+74/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_mk_nk_mn_instance.cpp`  (+73/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_mk_kn_mn_instance.cpp`  (+62/-0)
- `example/15_grouped_gemm/README.md`  (+58/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_km_kn_mn_instance.cpp`  (+53/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_km_nk_mn_instance.cpp`  (+53/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm.hpp`  (+29/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/CMakeLists.txt`  (+15/-0)
- `profiler/src/profiler.cpp`  (+10/-0)
- `profiler/CMakeLists.txt`  (+3/-0)

## Key added lines (kernel files)

**`example/15_grouped_gemm/grouped_gemm_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`include/ck/tensor_operation/gpu/device/device_gemm.hpp`**
```
struct GemmShape
ck::index_t M, N, K;
ck::index_t StrideA, StrideB, StrideC;
template <typename AElementwiseOperation,
```

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm_xdl.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ADataType,
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp`**
```
template <typename GridwiseGemm,
typename FloatAB,
typename FloatC,
typename GemmDesc,
```

**`library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_km_kn_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace device_grouped_gemm_instance {
```
