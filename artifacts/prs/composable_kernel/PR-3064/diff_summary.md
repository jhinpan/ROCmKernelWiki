# Diff summary

- **files changed:** 20
- **lines:** +1122 / -0
- **kernel-ish files:** 18

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`  (+472/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_reboot_util.hpp`  (+283/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_reboot_types.hpp`  (+56/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_reboot_smoke_cases.inc`  (+47/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_reboot_extended_cases.inc`  (+24/-0)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm_tile_partitioner_impl.hpp`  (+21/-0)
- `test/ck_tile/gemm_streamk/test_streamk_tile_partitioner.cpp`  (+20/-0)
- `test/ck_tile/gemm_streamk/extended_tests/test_gemm_streamk_reboot_bf16_nonpersistent.cpp`  (+19/-0)
- `test/ck_tile/gemm_streamk/extended_tests/test_gemm_streamk_reboot_bf16_persistent.cpp`  (+19/-0)
- `test/ck_tile/gemm_streamk/extended_tests/test_gemm_streamk_reboot_fp16_nonpersistent.cpp`  (+19/-0)
- `test/ck_tile/gemm_streamk/extended_tests/test_gemm_streamk_reboot_fp16_persistent.cpp`  (+19/-0)
- `test/ck_tile/gemm_streamk/smoke_tests/test_gemm_streamk_reboot_bf16_nonpersistent.cpp`  (+19/-0)
- `test/ck_tile/gemm_streamk/smoke_tests/test_gemm_streamk_reboot_bf16_persistent.cpp`  (+19/-0)
- `test/ck_tile/gemm_streamk/smoke_tests/test_gemm_streamk_reboot_fp16_nonpersistent.cpp`  (+19/-0)
- `test/ck_tile/gemm_streamk/smoke_tests/test_gemm_streamk_reboot_fp16_persistent.cpp`  (+19/-0)

## Key added lines (kernel files)

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`**
```
namespace reboot {
struct StreamKHostArgs : public ck_tile::UniversalGemmHostArgs<>
CK_TILE_HOST explicit StreamKHostArgs(const void* a_ptr_,
const void* b_ptr_,
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_tile_partitioner.hpp`**
```
CK_TILE_HOST index_t estimate_num_wgs_per_tile() const noexcept;
static constexpr bool PERSISTENT = true;
static constexpr bool PERSISTENT = false;
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_tile_partitioner_impl.hpp`**
```
template <typename BlockGemmShapeType, StreamKReductionStrategy ReductionStrategyType>
CK_TILE_HOST index_t
StreamKTilePartitionerBase<BlockGemmShapeType, ReductionStrategyType>::estimate_num_wgs_per_tile()
const noexcept
```

**`test/ck_tile/gemm_streamk/extended_tests/test_gemm_streamk_reboot_bf16_nonpersistent.cpp`**
```
template <typename Tuple>
class TestCkTileStreamKRebootBf16NonPersistent : public TestCkTileStreamKReboot<Tuple>
TYPED_TEST_SUITE(TestCkTileStreamKRebootBf16NonPersistent, KernelTypesStreamKBf16NonPersistent);
```

**`test/ck_tile/gemm_streamk/extended_tests/test_gemm_streamk_reboot_bf16_persistent.cpp`**
```
template <typename Tuple>
class TestCkTileStreamKRebootBf16Persistent : public TestCkTileStreamKReboot<Tuple>
TYPED_TEST_SUITE(TestCkTileStreamKRebootBf16Persistent, KernelTypesStreamKBf16Persistent);
```
