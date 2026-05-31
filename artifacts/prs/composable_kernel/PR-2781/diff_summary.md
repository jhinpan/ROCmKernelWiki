# Diff summary

- **files changed:** 10
- **lines:** +612 / -35
- **kernel-ish files:** 8

## Files (by churn)

- `test/ck_tile/gemm_streamk/test_gemm_streamk_util.hpp`  (+282/-0)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`  (+146/-11)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_cases.inc`  (+118/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_types.hpp`  (+25/-0)
- `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`  (+10/-10)
- `include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`  (+7/-12)
- `test/ck_tile/gemm_streamk/test_gemm_streamk.cpp`  (+14/-0)
- `test/ck_tile/gemm_streamk/CMakeLists.txt`  (+7/-0)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+2/-2)
- `test/ck_tile/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`**
```
uint32_t iter_end) const noexcept
uint32_t distance_to_tile_boundary =
k_iters_per_tile.get() - (iter_start % k_iters_per_tile.get());
return min(iter_start + distance_to_tile_boundary, iter_end) - iter_start;
```

**`include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`**
```
{a_ptr}, {b_ptr}, {/*ds_ptr*/}, c_ptr, kargs, splitk_batch_offset.splitted_k);
{a_ptr}, {b_ptr}, {/*ds_ptr*/}, c_ptr, kargs, splitk_batch_offset.splitted_k);
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`**
```
CK_TILE_HOST static StreamKKernelArgs MakeKernelArgs(const StreamKHostArgs& host_args,
int num_cu    = NumCU(),
int occupancy = Occupancy())
static_cast<uint32_t>(num_cu),
```

**`include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`**
```
const index_t k_size)
make_tuple(kargs.M, k_size),
make_tuple(k_size, kargs.M),
const index_t K0     = k_size / K1;
```

**`test/ck_tile/gemm_streamk/test_gemm_streamk.cpp`**
```
TYPED_TEST_SUITE(TestCkTileStreamK, KernelTypesStreamK);
```
