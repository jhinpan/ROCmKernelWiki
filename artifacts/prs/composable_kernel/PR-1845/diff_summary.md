# Diff summary

- **files changed:** 21
- **lines:** +602 / -92
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/core/arch/generic_memory_space_atomic.hpp`  (+293/-10)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+91/-8)
- `example/ck_tile/03_gemm/gemm_basic.hpp`  (+44/-7)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+32/-6)
- `example/ck_tile/03_gemm/script/smoke_test_basic.sh`  (+18/-17)
- `example/ck_tile/03_gemm/script/smoke_test_mem_pipeline.sh`  (+18/-17)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+23/-6)
- `include/ck_tile/host/check_err.hpp`  (+13/-7)
- `example/ck_tile/03_gemm/script/benchmark_basic_fp8.sh`  (+14/-0)
- `example/ck_tile/03_gemm/script/benchmark_mem_pipeline_bf16.sh`  (+13/-0)
- `example/ck_tile/03_gemm/script/benchmark_mem_pipeline_bf8.sh`  (+13/-0)
- `example/ck_tile/03_gemm/script/benchmark_mem_pipeline_fp8.sh`  (+13/-0)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+5/-5)
- `example/ck_tile/03_gemm/script/benchmark_mem_pipeline.sh`  (+3/-3)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+3/-2)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
template <typename ADataType,
typename BDataType,
typename AccDataType,
typename CDataType,
```

**`example/ck_tile/03_gemm/gemm_basic.hpp`**
```
template <>
struct GemmBasicTypeConfig<ck_tile::bf16_t>
using ADataType   = ck_tile::bf16_t;
using BDataType   = ck_tile::bf16_t;
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType,
typename ALayout, typename BLayout, typename CLayout>
float ave_time = gemm_calc<ADataType, BDataType, AccDataType, CDataType,
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
template <typename ADataType,
typename BDataType,
typename AccDataType,
typename CDataType,
```

**`include/ck_tile/core/arch/generic_memory_space_atomic.hpp`**
```
template <typename T, typename ComputeType>
CK_TILE_HOST_DEVICE T add(const T& a, const T& b)
return type_convert<T>(type_convert<ComputeType>(a) + type_convert<ComputeType>(b));
rtn[0] = add<bf16_t, float>(a[0], b[0]);
```
