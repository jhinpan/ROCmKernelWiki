# Diff summary

- **files changed:** 15
- **lines:** +1098 / -2
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+258/-0)
- `example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`  (+253/-0)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+225/-0)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+112/-0)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+103/-0)
- `example/ck_tile/16_batched_gemm/batched_gemm.hpp`  (+63/-0)
- `example/ck_tile/16_batched_gemm/README.md`  (+37/-0)
- `test/ck_tile/batched_gemm/test_batched_gemm.cpp`  (+29/-0)
- `test/ck_tile/batched_gemm/test_batched_gemm_ut_cases.inc`  (+9/-0)
- `test/ck_tile/batched_gemm/CMakeLists.txt`  (+4/-0)
- `example/ck_tile/CMakeLists.txt`  (+1/-1)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+1/-1)
- `example/ck_tile/16_batched_gemm/CMakeLists.txt`  (+1/-0)
- `include/ck_tile/ops/gemm.hpp`  (+1/-0)
- `test/ck_tile/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
template <typename ALayout, typename BLayout, typename CLayout>
float batched_gemm(const batched_gemm_kargs& args, const ck_tile::stream_config& s)
constexpr bool kPadM        = false;
constexpr bool kPadN        = false;
```

**`example/ck_tile/16_batched_gemm/batched_gemm.hpp`**
```
template <typename DataType>
struct BatchedGemmTypeConfig;
template <>
struct BatchedGemmTypeConfig<ck_tile::half_t>
```

**`example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`**
```
template <typename ALayout, typename BLayout, typename CLayout>
float invoke_batched_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
ck_tile::DeviceMem& b_k_n_dev_buf,
ck_tile::DeviceMem& c_m_n_dev_buf,
```

**`include/ck_tile/host/reference/reference_gemm.hpp`**
```
template <typename ADataType,
typename BDataType,
typename AccDataType,
typename CDataType,
```

**`include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`**
```
namespace ck_tile {
struct BatchedGemmHostArgs
const void* a_ptr;
const void* b_ptr;
```
