# Diff summary

- **files changed:** 29
- **lines:** +1655 / -556
- **kernel-ish files:** 26

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`  (+413/-0)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+45/-321)
- `test/ck_tile/gemm/test_gemm_mem_pipeline_util.hpp`  (+318/-0)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+217/-0)
- `example/ck_tile/03_gemm/gemm_mem_pipeline.cpp`  (+188/-0)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+62/-61)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp`  (+71/-0)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+21/-39)
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`  (+29/-29)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+44/-9)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`  (+26/-26)
- `test/ck_tile/gemm/test_gemm_mem_pipeline_ut_cases.inc`  (+41/-0)
- `example/ck_tile/03_gemm/gemm_basic.hpp`  (+26/-6)
- `include/ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp`  (+15/-15)
- `test/ck_tile/gemm/test_gemm_mem_pipeline.cpp`  (+29/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
template <typename ALayout, typename BLayout, typename CLayout>
constexpr bool kPadC        = true;
constexpr ck_tile::index_t kOutputRank = 2;
constexpr ck_tile::index_t M_Tile = 128;
```

**`example/ck_tile/03_gemm/gemm_basic.hpp`**
```
using CDataType   = ck_tile::half_t;
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("b", "1", "batch size")
```

**`example/ck_tile/03_gemm/gemm_mem_pipeline.cpp`**
```
template <typename ALayout, typename BLayout, typename CLayout>
float gemm_calc(const gemm_basic_args& args, const ck_tile::stream_config& s)
constexpr ck_tile::index_t M_Tile = 128;
constexpr ck_tile::index_t N_Tile = 128;
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
template <typename ALayout, typename BLayout, typename CLayout>
float invoke_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
ck_tile::DeviceMem& b_k_n_dev_buf,
ck_tile::DeviceMem& c_m_n_dev_buf,
```

**`include/ck_tile/core/tensor/load_tile.hpp`**
```
template <typename DistributedTensor_,
typename BottomTensorView_,
typename WindowLengths_,
typename TileDistribution_,
```
