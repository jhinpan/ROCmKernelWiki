# Diff summary

- **files changed:** 16
- **lines:** +12 / -13
- **kernel-ish files:** 2

## Files (by churn)

- `csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`  (+1/-10)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv`  (+5/-1)
- `hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv`  (+4/-0)
- `op_tests/test_gemm_a8w8_blockscale.py`  (+2/-2)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_128x128.co`  (+0/-0)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_32x128.co`  (+0/-0)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_48x128.co`  (+0/-0)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_64x128.co`  (+0/-0)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_80x128.co`  (+0/-0)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_96x128.co`  (+0/-0)
- `hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_128x128.co`  (+0/-0)
- `hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_32x128.co`  (+0/-0)
- `hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_48x128.co`  (+0/-0)
- `hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_64x128.co`  (+0/-0)
- `hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_80x128.co`  (+0/-0)

## Key added lines (kernel files)

**`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`**
```
int Ndim = B.size(0), Kdim = A.size(1);
```

**`op_tests/test_gemm_a8w8_blockscale.py`**
```
weight_asm = shuffle_weight(weight, layout=(16, 16))
c, avg_c = run_asm(x, weight_asm, x_scale_t, w_scale, dtype)
```
