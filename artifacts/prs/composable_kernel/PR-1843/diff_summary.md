# Diff summary

- **files changed:** 10
- **lines:** +283 / -86
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`  (+237/-46)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+21/-20)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+6/-6)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+5/-6)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+5/-2)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+5/-2)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+1/-1)
- `example/ck_tile/03_gemm/gemm_basic.hpp`  (+1/-1)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+1/-1)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;
```

**`example/ck_tile/03_gemm/gemm_basic.hpp`**
```
.insert("b_layout", "C", "B tensor data layout - Column by default")
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
constexpr int kBlockPerCu                         = 1;
constexpr ck_tile::index_t TileParitionerGroupNum = 8;
constexpr ck_tile::index_t TileParitionerM01      = 4;
using TilePartitioner = ck_tile::
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;
```

**`include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`**
```
return dim3(TilePartitioner::GridSize(M, N), batch_count, KBatch);
const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockIdx.x);
const auto i_batch  = __builtin_amdgcn_readfirstlane(blockIdx.y);
const auto i_splitk = __builtin_amdgcn_readfirstlane(blockIdx.z);
```
