# Diff summary

- **files changed:** 7
- **lines:** +641 / -304
- **kernel-ish files:** 7

## Files (by churn)

- `kernels/moe_gemm_2stage.py`  (+230/-123)
- `kernels/preshuffle_gemm.py`  (+216/-86)
- `tests/kernels/test_moe_gemm.py`  (+56/-37)
- `kernels/mfma_preshuffle_pipeline.py`  (+67/-21)
- `tests/kernels/test_preshuffle_gemm.py`  (+44/-29)
- `tests/kernels/test_ref.py`  (+12/-8)
- `flydsl/src/flydsl/dialects/ext/rocdl.py`  (+16/-0)

## Key added lines (kernel files)

**`flydsl/src/flydsl/dialects/ext/rocdl.py`**
```
_ods_mfma_f32_16x16x16bf16_1k = globals().get("mfma_f32_16x16x16bf16_1k", None)
def mfma_f32_16x16x16bf16_1k_op(result_type, operands, *, loc=None, ip=None):
"""Return the op view (original behavior)."""
if _ods_mfma_f32_16x16x16bf16_1k is None:
```

**`kernels/mfma_preshuffle_pipeline.py`**
```
def make_preshuffle_b_layout(
c_n: ir.Value,
c_k: ir.Value,
kpack_bytes: int = 16,
```

**`kernels/moe_gemm_2stage.py`**
```
"""MoE GEMM stage1/stage2 kernel implementations (FLIR MFMA FP8/FP16).
- "fp16": X/W are fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
if in_dtype not in ("fp8", "fp16", "int8", "int4"):
raise ValueError(f"in_dtype must be one of ('fp8','fp16','int8','int4'), got {in_dtype!r}")
```

**`kernels/preshuffle_gemm.py`**
```
if in_dtype not in ("fp8", "int8", "int4", "fp16", "bf16"):
raise ValueError(
"in_dtype must be one of ('fp8','int8','int4','fp16','bf16'), "
f"got {in_dtype!r}"
```

**`tests/kernels/test_moe_gemm.py`**
```
if in_dtype not in ("fp8", "fp16", "int8", "int4"):
raise ValueError(f"in_dtype must be one of ('fp8','fp16','int8','int4'), got {in_dtype!r}")
elif in_dtype == "fp16":
x_q = x_fp32.to(torch.float16)
```
