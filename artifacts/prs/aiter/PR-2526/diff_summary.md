# Diff summary

- **files changed:** 32
- **lines:** +106 / -45
- **kernel-ish files:** 1

## Files (by churn)

- `csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`  (+100/-39)
- `hsa/gfx950/i8gemm/i8gemm_bf16_perTokenI8.csv`  (+3/-3)
- `hsa/gfx942/i8gemm/i8gemm_bf16_perTokenI8.csv`  (+2/-2)
- `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv`  (+1/-1)
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

## Key added lines (kernel files)

**`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`**
```
std::tuple<std::string, int> get_heuristic_fp8_kernel(
std::string arch_id,
std::optional<int> k_split,
std::optional<bool> bpreshuffle,
```
