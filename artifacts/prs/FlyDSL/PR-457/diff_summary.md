# Diff summary

- **files changed:** 12
- **lines:** +173 / -137
- **kernel-ish files:** 12

## Files (by churn)

- `kernels/flash_attn_func.py`  (+23/-38)
- `kernels/moe_gemm_2stage_mxscale_gfx1250.py`  (+36/-10)
- `kernels/preshuffle_gemm_v2.py`  (+16/-12)
- `kernels/gemm_fp8fp4_gfx1250.py`  (+13/-13)
- `kernels/moe_blockscale_2stage.py`  (+6/-18)
- `python/flydsl/compiler/mlir_utils.py`  (+24/-0)
- `kernels/wmma_gemm_gfx1250.py`  (+11/-12)
- `python/flydsl/compiler/kernel_function.py`  (+20/-3)
- `kernels/blockscale_preshuffle_gemm.py`  (+10/-10)
- `kernels/preshuffle_gemm.py`  (+11/-9)
- `kernels/moe_gemm_2stage_common_gfx1250.py`  (+1/-10)
- `kernels/moe_gemm_2stage_wmma_gfx1250.py`  (+2/-2)

## Key added lines (kernel files)

**`kernels/blockscale_preshuffle_gemm.py`**
```
kernel_gemm(
arg_scale_a,
arg_scale_b,
value_attrs={"rocdl.waves_per_eu": waves_per_eu},
```

**`kernels/flash_attn_func.py`**
```
passthrough_entries = (
["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
["no-nans-fp-math", "true"],
["unsafe-fp-math", "true"],
```

**`kernels/gemm_fp8fp4_gfx1250.py`**
```
kernel_mxscale_gemm(
arg_a_scale,
arg_b_scale,
value_attrs={
```

**`kernels/moe_blockscale_2stage.py`**
```
moe_blockscale_gemm1(
value_attrs={"rocdl.waves_per_eu": waves_per_eu},
).launch(grid=(gx, gy, 1), block=(256, 1, 1), stream=stream)
moe_blockscale_gemm2(
```

**`kernels/moe_gemm_2stage_common_gfx1250.py`**
```
def _finalize_alloc_and_launch_2d(*, ctx, alloc, launcher, gx, gy, block_threads: int, stream, ir,
```
