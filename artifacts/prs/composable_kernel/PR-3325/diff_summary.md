# Diff summary

- **files changed:** 11
- **lines:** +58 / -46
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+22/-26)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp`  (+11/-12)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+7/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+3/-3)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+6/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp`  (+4/-0)
- `example/ck_tile/38_block_scale_gemm/README.md`  (+1/-1)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8.cpp`  (+1/-1)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8i4.cpp`  (+1/-1)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8.cpp`  (+1/-1)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8i4.cpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8.cpp`**
```
using GemmConfig = GemmConfigQuantPrefill<T>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8i4.cpp`**
```
using GemmConfig = GemmConfigQuantPrefill<T>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8.cpp`**
```
using GemmConfig = GemmConfigQuantPrefill<T>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8i4.cpp`**
```
using GemmConfig = GemmConfigQuantPrefill<T>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
struct GemmConfigQuantPrefill : public GemmConfigBase
struct GemmConfigPreshuffleBQuantPrefill : public GemmConfigQuantPrefill<PrecType>
struct GemmConfigBQuantPrefill_Wmma : public GemmConfigQuantPrefill<PrecType>
```
