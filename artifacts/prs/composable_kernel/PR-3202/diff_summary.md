# Diff summary

- **files changed:** 147 (diff was byte-capped; summary is partial)
- **lines:** +580 / -3490
- **kernel-ish files:** 145

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`  (+175/-523)
- `include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`  (+0/-444)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_reboot_util.hpp`  (+0/-287)
- `test/ck_tile/gemm_streamk/test_gemm_streamk.hpp`  (+0/-282)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_cases.inc`  (+0/-174)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_types.hpp`  (+65/-105)
- `test/ck_tile/gemm_streamk/CMakeLists.txt`  (+21/-139)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_types_bf16.hpp`  (+0/-76)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_reboot_types.hpp`  (+0/-56)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm_tile_partitioner_impl.hpp`  (+16/-22)
- `include/ck_tile/ops/common/streamk_common.hpp`  (+0/-29)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_types_bf8.hpp`  (+0/-25)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm_tile_partitioner.hpp`  (+11/-11)
- `example/ck_tile/40_streamk_gemm/run_gemm_example.inc`  (+10/-10)
- `test/ck_tile/gemm_streamk/extended_tests/test_gemm_streamk_reboot_bf16_nonpersistent.cpp`  (+0/-19)

## Key added lines (kernel files)

**`example/ck_tile/40_streamk_gemm/run_gemm_example.inc`**
```
ck_tile::StreamKHostArgs args{a_m_k_dev_buf.GetDeviceBuffer(),
b_k_n_dev_buf.GetDeviceBuffer(),
c_m_n_dev_buf.GetDeviceBuffer(),
stride_A,
```

**`example/ck_tile/40_streamk_gemm/streamk_gemm_basic.cpp`**
```
std::tuple<float, ck_tile::index_t> gemm(const ck_tile::StreamKHostArgs& args,
ck_tile::StreamKTilePartitioner<GemmShape, ReductionStrategy, GemmConfig::Persistent>;
using Kernel = ck_tile::StreamKKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`**
```
CK_TILE_DEVICE
void StreamKGemm(StreamKKernelArgs& kargs, index_t cta_idx, void* smem_ptr_0) const
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_tile_partitioner.hpp`**
```
struct StreamKTilePartitioner;
struct StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, true>
StreamKTilePartitioner(ck_tile::index_t m,
ck_tile::index_t n,
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_tile_partitioner_impl.hpp`**
```
struct StreamKTilePartitioner;
StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, true>::StreamKTilePartitioner(
ck_tile::index_t m, ck_tile::index_t n, ck_tile::index_t k, ck_tile::index_t grid)
StreamKTilePartitioner<BlockGemmShapeType, ReductionStrategyType, true>::grid_size() const noexcept
```
