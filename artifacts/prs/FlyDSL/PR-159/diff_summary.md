# Diff summary

- **files changed:** 12
- **lines:** +83 / -83
- **kernel-ish files:** 9

## Files (by churn)

- `tests/kernels/test_moe_gemm.py`  (+28/-28)
- `kernels/mixed_moe_gemm_2stage.py`  (+11/-11)
- `kernels/mixed_preshuffle_gemm.py`  (+9/-9)
- `kernels/preshuffle_gemm.py`  (+8/-8)
- `kernels/mfma_preshuffle_pipeline.py`  (+7/-7)
- `tests/kernels/test_preshuffle_gemm.py`  (+7/-7)
- `kernels/moe_gemm_2stage.py`  (+4/-4)
- `docs/prebuilt_kernels_guide.md`  (+3/-3)
- `kernels/mfma_epilogues.py`  (+3/-3)
- `docs/layout_system_guide.md`  (+1/-1)
- `flir/include/flir/FlirOps.td`  (+1/-1)
- `flir/lib/Transforms/FlirToStandard.cpp`  (+1/-1)

## Key added lines (kernel files)

**`kernels/mfma_epilogues.py`**
```
LDS CShuffle epilogue based on input parameters.
A LDS CShuffle epilogue skeleton:
"""LDS CShuffle epilogue skeleton.
```

**`kernels/mfma_preshuffle_pipeline.py`**
```
"""Build B preshuffle layout for A8 MFMA kernels.
"""Build scale preshuffle layout for MXFP4 MFMA kernels.
"""Store one 16B chunk into LDS with XOR16 swizzle on the K dimension."""
"""Store one 8B chunk into LDS with XOR16 swizzle on the K dimension."""
```

**`kernels/mixed_preshuffle_gemm.py`**
```
- `ck_v1_single_lds`: Intrawave + bpreshuffle v1 spirit (single LDS buffer for A)
"""Intrawave scheduler adapted from preshuffle_gemm.py.
```

**`kernels/preshuffle_gemm.py`**
```
- `ck_v1_single_lds`: Intrawave + bpreshuffle v1 spirit (single LDS buffer for A)
```

**`tests/kernels/test_moe_gemm.py`**
```
"""Build routing buffers once, reusable across stage1 + stage2.
pytest.skip("aiter not available; cannot compare to aiter moe stage1.", allow_module_level=False)
print(f"[aiter] stage1: {us_ck:.1f} us, {tflops_ck:.2f} TFLOPS, flir vs aiter speedups: {tflops / tflops_ck:.2f}x")
logging.warning(f"Skipping aiter moe stage1 compare (not runnable here): {e}")
```
