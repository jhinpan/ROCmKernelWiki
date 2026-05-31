# Diff summary

- **files changed:** 8
- **lines:** +33 / -41
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+20/-23)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+6/-7)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp`  (+0/-7)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+7/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+0/-1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+0/-1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_fp8.hpp`  (+0/-1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qs_ks_vs.hpp`  (+0/-1)

## Key added lines (kernel files)

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`**
```
using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`**
```
else if constexpr(std::is_same_v<typename Problem::QDataType, fp8_t> &&
std::is_same_v<typename Problem::KDataType, fp8_t> &&
std::is_same_v<typename Problem::SaccDataType, float>)
constexpr index_t swizzle_factor = 4;
```

**`include/ck_tile/ops/gemm/warp/warp_gemm.hpp`**
```
template <index_t swizzle_factor = 2>
using WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution =
WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<fp8_t, fp8_t>,
```
