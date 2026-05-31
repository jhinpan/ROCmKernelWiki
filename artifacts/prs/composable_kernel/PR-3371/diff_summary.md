# Diff summary

- **files changed:** 13
- **lines:** +525 / -71
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_kernel.hpp`  (+144/-53)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_reduction_cases.inc`  (+88/-0)
- `test/ck_tile/gemm_streamk/test_streamk_tile_partitioner.cpp`  (+79/-0)
- `test/ck_tile/gemm_streamk/test_streamk_tile_partitioner_common.hpp`  (+67/-1)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner_impl.hpp`  (+33/-3)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_coherency.hpp`  (+35/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_util.hpp`  (+23/-8)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner.hpp`  (+21/-1)
- `test/ck_tile/gemm_streamk/smoke_tests/test_gemm_streamk_fp16_reduction.cpp`  (+17/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_types.hpp`  (+8/-0)
- `include/ck_tile/ops/gemm.hpp`  (+4/-3)
- `include/ck_tile/ops/common/streamk_common.hpp`  (+3/-2)
- `test/ck_tile/gemm_streamk/CMakeLists.txt`  (+3/-0)

## Key added lines (kernel files)

**`include/ck_tile/ops/common/streamk_common.hpp`**
```
Atomic        = 0u,
Reduction     = 1u,
TreeReduction = 2u
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_coherency.hpp`**
```
namespace ck_tile {
template <typename CompilerTarget, typename Enabler = void>
struct StreamKCoherency
static constexpr amd_buffer_coherence_enum BUFFER_COHERENCE =
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_kernel.hpp`**
```
auto* sk_flags_ptr = static_cast<index_t*>(kargs.workspace_ptr);
index_t offset     = cta_idx * sizeof(index_t);
asm volatile("s_mov_b32 m0, %2\n\t"
"s_store_dword %0, %1, %2 glc\n\t"
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner.hpp`**
```
CK_TILE_DEVICE index_t get_start_iter(index_t cta_idx) const noexcept;
CK_TILE_DEVICE index_t get_tile_local_cta_index(index_t tile_iter_start,
index_t cta_idx) const noexcept;
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner_impl.hpp`**
```
template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_DEVICE index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::get_start_iter(
index_t cta_idx) const noexcept
```
