# Diff summary

- **files changed:** 19
- **lines:** +3619 / -21
- **kernel-ish files:** 18

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_fpAintB_gemm_wmma.hpp`  (+1200/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_fpAintB_gemm_wmma.hpp`  (+713/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_fpAintB_gemm_wmma.hpp`  (+624/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp`  (+219/-0)
- `example/49_fpAintB_gemm/run_gemm_example.inc`  (+187/-0)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_fpAintB_gemm.hpp`  (+177/-0)
- `example/49_fpAintB_gemm/common.hpp`  (+123/-0)
- `include/ck/utility/amd_buffer_addressing.hpp`  (+110/-1)
- `example/49_fpAintB_gemm/fp16int8_gemm_wmma.cpp`  (+93/-0)
- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+77/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_dequantB.hpp`  (+46/-0)
- `include/ck/utility/data_type.hpp`  (+26/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_wmma_f16_f16_f16_km_kn_mn_instance.cpp`  (+4/-5)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_wmma_f16_f16_f16_km_nk_mn_instance.cpp`  (+4/-5)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_wmma_f16_f16_f16_mk_kn_mn_instance.cpp`  (+3/-4)

## Key added lines (kernel files)

**`example/49_fpAintB_gemm/common.hpp`**
```
struct ProblemSize final
ck::index_t M = 3840;
ck::index_t N = 4096;
ck::index_t K = 4096;
```

**`example/49_fpAintB_gemm/fp16int8_gemm_wmma.cpp`**
```
using ADataType        = ck::half_t;
using QuantDataType    = int8_t;
using BDataType        = uint8_t;
using ScaleDataType    = ck::half_t;
```

**`example/49_fpAintB_gemm/run_gemm_example.inc`**
```
bool run_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
using namespace ck::literals;
auto& [M, N, K, StrideA, StrideB, StrideC] = problem_size;
```

**`include/ck/tensor_operation/gpu/block/blockwise_fpAintB_gemm_wmma.hpp`**
```
namespace ck {
template <index_t BlockSize,
typename ADataType,
typename BDataType,
```

**`include/ck/tensor_operation/gpu/device/device_gemm_dequantB.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```
