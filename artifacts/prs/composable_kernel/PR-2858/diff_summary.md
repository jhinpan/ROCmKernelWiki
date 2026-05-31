# Diff summary

- **files changed:** 13
- **lines:** +183 / -177
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck_tile/ops/common/load_interleaved_pk_type.hpp`  (+58/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_smoke_run_test.inc`  (+2/-55)
- `include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`  (+11/-26)
- `test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_util.hpp`  (+29/-7)
- `include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`  (+8/-23)
- `include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+8/-22)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v1.hpp`  (+17/-11)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp`  (+17/-11)
- `test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_kernel_types.hpp`  (+15/-10)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp`  (+11/-7)
- `test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_ut_cases.inc`  (+4/-4)
- `test/ck_tile/batched_gemm/test_batched_gemm_ut_cases.inc`  (+2/-1)
- `CHANGELOG.md`  (+1/-0)

## Key added lines (kernel files)

**`include/ck_tile/ops/common/load_interleaved_pk_type.hpp`**
```
namespace ck_tile {
template <class T>
struct is_pk_int4 : std::false_type
template <>
```

**`include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`**
```
template <typename Problem_,
typename Policy_     = BlockGemmASmemBSmemCRegV1DefaultPolicy,
index_t UnaryOpSize_ = 8>
using Loader   = remove_cvref_t<InterleavedPKTypeLoader<ComputeDataType, UnaryOpSize_>>;
```

**`include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp`**
```
using BTypeToUse =
std::conditional_t<std::is_same_v<typename Problem::BDataType, ck_tile::pk_int4_t>,
typename Problem::ADataType,
typename Problem::BDataType>;
```

**`include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v1.hpp`**
```
bool>* = nullptr,
index_t UnaryOpSize_             = 8>
using BTypeToUse =
std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
```

**`include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp`**
```
bool>* = nullptr,
index_t UnaryOpSize_             = 8>
using BTypeToUse =
std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
```
