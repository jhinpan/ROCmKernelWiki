# Diff summary

- **files changed:** 4
- **lines:** +24 / -0
- **kernel-ish files:** 4

## Files (by churn)

- `aiter/jit/core.py`  (+7/-0)
- `aiter/jit/utils/chip_info.py`  (+7/-0)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`  (+5/-0)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py`  (+5/-0)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
"gfx1101",
"gfx1102",
"gfx1103",
"gfx1150",
```

**`aiter/jit/utils/chip_info.py`**
```
9: "gfx1101",
10: "gfx1102",
11: "gfx1103",
12: "gfx1150",
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`**
```
"gfx1103",
"gfx1150",
"gfx1151",
"gfx1152",
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py`**
```
"gfx1103",
"gfx1150",
"gfx1151",
"gfx1152",
```
