# Diff summary

- **files changed:** 18
- **lines:** +758 / -92
- **kernel-ish files:** 15

## Files (by churn)

- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+274/-0)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+176/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp`  (+45/-40)
- `example/ck_tile/03_gemm/gemm_basic.hpp`  (+71/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+27/-24)
- `include/ck_tile/ops/gemm/pipeline/block_gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`  (+37/-6)
- `include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`  (+38/-0)
- `example/ck_tile/03_gemm/README.md`  (+23/-0)
- `include/ck_tile/ops/gemm/pipeline/block_gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+13/-5)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+13/-4)
- `include/ck_tile/ops/gemm/pipeline/block_gemm_pipeline_problem.hpp`  (+14/-3)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp`  (+10/-4)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_bgmem_creg_v1.hpp`  (+6/-5)
- `include/ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp`  (+4/-0)
- `include/ck_tile/ops/gemm/pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp`  (+2/-1)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("b", "1", "batch size")
.insert("m", "1024", "m dimension")
```

**`example/ck_tile/03_gemm/gemm_basic.hpp`**
```
template <typename DataType>
struct GemmBasicTypeConfig;
template <>
struct GemmBasicTypeConfig<ck_tile::half_t>
```

**`include/ck_tile/host/reference/reference_gemm.hpp`**
```
typename LayoutA,
typename LayoutB,
typename LayoutC,
const int K = (std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp`**
```
using BlockGemmProblem = BlockGemmPipelineProblem<
typename Problem::QDataType,
typename Problem::KDataType,
typename Problem::AccDataType,
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`**
```
using BlockGemmProblem = BlockGemmPipelineProblem<
typename Problem::QDataType,
typename Problem::KDataType,
typename Problem::SaccDataType,
```
