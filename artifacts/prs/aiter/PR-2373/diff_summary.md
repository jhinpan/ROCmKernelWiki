# Diff summary

- **files changed:** 30
- **lines:** +12 / -3
- **kernel-ish files:** 1

## Files (by churn)

- `hsa/gfx950/i8gemm/i8gemm_bf16_perTokenI8.csv`  (+9/-0)
- `csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`  (+3/-3)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_128x128.co`  (+0/-0)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_32x128.co`  (+0/-0)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_48x128.co`  (+0/-0)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_64x128.co`  (+0/-0)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_80x128.co`  (+0/-0)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_96x128.co`  (+0/-0)
- `hsa/gfx942/i8gemm/I8gemm_bf16_perTokenI8_BpreShuffle_128x128.co`  (+0/-0)
- `hsa/gfx942/i8gemm/I8gemm_bf16_perTokenI8_BpreShuffle_160x128.co`  (+0/-0)
- `hsa/gfx942/i8gemm/I8gemm_bf16_perTokenI8_BpreShuffle_16x128.co`  (+0/-0)
- `hsa/gfx942/i8gemm/I8gemm_bf16_perTokenI8_BpreShuffle_32x128.co`  (+0/-0)
- `hsa/gfx942/i8gemm/I8gemm_bf16_perTokenI8_BpreShuffle_48x128.co`  (+0/-0)
- `hsa/gfx942/i8gemm/I8gemm_bf16_perTokenI8_BpreShuffle_64x128.co`  (+0/-0)
- `hsa/gfx942/i8gemm/I8gemm_bf16_perTokenI8_BpreShuffle_80x128.co`  (+0/-0)

## Key added lines (kernel files)

**`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`**
```
if (cfg.bpreshuffle == bpreshuffle_en && ((cfg.splitK >= splitK_en) || !splitK.has_value())) {
std::vector<int> splitK_list = (splitK.has_value())
printf("\n=== A8W8 GEMM Kernel Parameters ===\n");
```
