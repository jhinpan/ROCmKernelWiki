# Diff summary

- **files changed:** 11
- **lines:** +844 / -61
- **kernel-ish files:** 8

## Files (by churn)

- `test/ck_tile/gemm_persistent_async_input/test_gemm_persistent_async_input.cpp`  (+304/-0)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+195/-34)
- `example/ck_tile/03_gemm/universal_gemm_invoker.hpp`  (+170/-0)
- `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`  (+72/-26)
- `include/ck_tile/core/utility/persistent_async_input_scheduler.hpp`  (+49/-0)
- `include/ck_tile/core/arch/workgroup_barrier.hpp`  (+30/-0)
- `test/ck_tile/gemm_persistent_async_input/CMakeLists.txt`  (+19/-0)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+2/-1)
- `CHANGELOG.md`  (+1/-0)
- `include/ck_tile/core.hpp`  (+1/-0)
- `test/ck_tile/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
.insert("rotating_count", "1000", "rotating count, defaults to 1000")
.insert("test_async", "0", "0: normal gemm, 1: test async input scheduler");
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
template <typename GemmConfig,
typename ADataType,
typename BDataType = ADataType,
typename CDataType = ADataType,
```

**`example/ck_tile/03_gemm/universal_gemm_invoker.hpp`**
```
template <typename GemmConfig,
typename ADataType,
typename BDataType,
typename DsDataType,
```

**`include/ck_tile/core/arch/workgroup_barrier.hpp`**
```
CK_TILE_DEVICE void wait_eq_wave(uint32_t value, uint32_t offset = 0)
const uint32_t wave_size = static_cast<uint32_t>(warpSize);
if(threadIdx.x < wave_size)
uint32_t loaded_value = 0;
```

**`include/ck_tile/core/utility/persistent_async_input_scheduler.hpp`**
```
namespace ck_tile {
struct PersistentAsyncInputScheduler
uint32_t tiles_per_chunk_m = 0;
uint32_t* chunk_signals = nullptr;
```
