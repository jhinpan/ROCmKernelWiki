# Diff summary

- **files changed:** 13
- **lines:** +264 / -154
- **kernel-ish files:** 11

## Files (by churn)

- `example/ck_tile/01_fmha/fmha_fwd_runner.hpp`  (+105/-65)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+29/-23)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+31/-17)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+36/-0)
- `example/ck_tile/01_fmha/example_fmha_fwd.cpp`  (+11/-19)
- `example/ck_tile/01_fmha/script/smoke_test_fwd.sh`  (+27/-2)
- `test/ck_tile/fmha/test_fmha_fwd.inc`  (+9/-12)
- `test/ck_tile/fmha/test_fmha_fwd_fp8.cpp`  (+6/-7)
- `example/ck_tile/01_fmha/codegen/ops/fmha_pagedkv_prefill.py`  (+5/-5)
- `example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`  (+2/-1)
- `test/ck_tile/batched_gemm/test_batched_gemm_ut_cases.inc`  (+2/-1)
- `example/ck_tile/01_fmha/README.md`  (+1/-1)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+0/-1)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
"fp8bf16": "FmhaFwdFp8Bf16",
"fp8fp32": "FmhaFwdFp8Fp32"
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
elif dtype == 'fp8' or dtype == 'fp8bf16':
elif dtype == 'fp8fp32':
(128,128) : [FmhaFwdTileSize(128, 128, 32, 128, 32,  128,  4, 1, 1,  4, 1, 1,  32, 32, 32,  32, 32, 32,  -1)],
squant = 'f'
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_pagedkv_prefill.py`**
```
for logits, mask, bias,  pagedkv, skip in itertools.product(["t", "f"], get_mask_map(mask_impl).keys(), BIAS_MAP.keys(),
for logits, mask, bias in itertools.product(["t", "f"], get_mask_map(mask_impl).keys(), BIAS_MAP.keys()):
pipelines.append(FmhaFwdPipeline('qr_pagedkv', 'row', 'f', 'f', 'f', 'f', logits, bias, 'f', 't', squant, mask, 'f'))
pipelines.append(FmhaFwdPipeline('qr_pagedkv', 'row', 't', 't', 'f', 'f', logits, bias, 'f', 't', squant, mask, 'f'))
```

**`example/ck_tile/01_fmha/example_fmha_fwd.cpp`**
```
"note when squant=1, this value will be modified")
"calculate scale_s, scale_p, scale_o auto")
"\n  tf or 2 - trig float\n")
else if(data_type == "fp8bf16")
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
struct FmhaFwdFp8Fp32
template <>
struct FmhaFwdTypeConfig<FmhaFwdFp8Bf16>
using QDataType             = ck_tile::fp8_t;
```
