# Diff summary

- **files changed:** 31
- **lines:** +916 / -482
- **kernel-ish files:** 28

## Files (by churn)

- `include/ck_tile/ops/fmha/block/block_dropout.hpp`  (+315/-386)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`  (+130/-1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+94/-32)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+61/-13)
- `test/ck_tile/fmha/test_fmha_fwd_fp32.cpp`  (+39/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+32/-1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs_default_policy.hpp`  (+21/-4)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_whole_k_prefetch_default_policy.hpp`  (+21/-4)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp`  (+13/-11)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+21/-1)
- `test/ck_tile/fmha/test_fmha_bwd_fp32.cpp`  (+20/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+18/-0)
- `include/ck_tile/core/utility/philox_rand.hpp`  (+8/-8)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+14/-0)
- `example/ck_tile/01_fmha/fmha_bwd_runner.hpp`  (+12/-2)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
"fp32"   : "FmhaFwdFp32",
"fp32": "FmhaBwdFp32",
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py`**
```
if receipt == 800 or receipt == 801:
cond = dtype == 'fp32'
if not cond:
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
if dtype == 'fp32' and tr_load == 'f':
FmhaBwdDQDKDVTileSize( 32, 128,  32,  32,  32,  32,  64,   32,   32, 1, 4, 1, 4, 1, 1, 2, 2, 1, 16, 16, 16, 16, 16, 16, 
FmhaBwdDQDKDVTileSize( 16,  64,  64,  16,  64,  16,  16,   64,   64, 1, 4, 1, 4, 1, 1, 1, 4, 1, 16, 16, 16, 16, 16, 16, 
FmhaBwdDQDKDVTileSize( 16,  64, 128,  16, 128,  16,  16,  128,  128, 1, 4, 1, 4, 1, 1, 1, 4, 1, 16, 16, 16, 16, 16, 16, 
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
[[maybe_unused]] const bool has_load_tr = ck_tile::is_load_tr_supported();
def seqtune(self, max_bm0 : int) -> str:
if self.bm0 == max_bm0: return 'true/*fall back to largest tile*/'
max_bm0 = max((t.bm0 for t in traits), default=0)
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_appendkv.py`**
```
if not per_dtypes:
per_dtypes += '    (void)t ; (void)s ; (void)a;'
if receipt == 800 or receipt == 801:
cond = dtype == 'fp32'
```
