# Diff summary

- **files changed:** 10
- **lines:** +22 / -5
- **kernel-ish files:** 10

## Files (by churn)

- `script/run_ck_profiler_gemm_with_csv_shapes.py`  (+6/-2)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+3/-0)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+3/-0)
- `include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`  (+2/-1)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+1/-1)
- `include/ck_tile/ops/gemm.hpp`  (+1/-1)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+2/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+2/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5.hpp`  (+1/-0)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
ck_tile::tuple<>,
ck_tile::tuple<>,
ck_tile::element_wise::PassThrough,
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
```

**`example/ck_tile/18_flatmm/flatmm_basic.cpp`**
```
ck_tile::tuple<>,
ck_tile::tuple<>,
ck_tile::element_wise::PassThrough,
```

**`include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`**
```
const auto& d_block_window      = gemm_tile_windows.at(I2);
c_block_window, c_block_tile, d_block_window, smem_ptr);
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`**
```
static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
```
