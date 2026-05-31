# Diff summary

- **files changed:** 7 (diff was byte-capped; summary is partial)
- **lines:** +1693 / -1448
- **kernel-ish files:** 5

## Files (by churn)

- `hsa/gfx942/fmoe_2stages/tune.py`  (+1035/-875)
- `aiter/configs/a8w8_tuned_gemm.csv`  (+482/-481)
- `aiter/utility/base_tuner.py`  (+97/-24)
- `aiter/configs/tuned_fmoe.csv`  (+57/-57)
- `aiter/fused_moe.py`  (+16/-5)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`  (+5/-5)
- `aiter/utility/mp_tuner.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
return metadata.stage1(
kernelName: str = "",
kernelName,
if run_1stage:
```

**`aiter/utility/base_tuner.py`**
```
return self.dtype2bpe_dict[dtype]
self.parser.add_argument(
action="store_true",
required=False,
```

**`aiter/utility/mp_tuner.py`**
```
return info, us, round(max_err_ratio, 4)
```

**`hsa/gfx942/fmoe_2stages/tune.py`**
```
class FmoeTuner(TunerCommon):
ARG_DEFAULTS = {
"verbose": False,
"tune_file": "aiter/configs/tuned_fmoe.csv",
```
