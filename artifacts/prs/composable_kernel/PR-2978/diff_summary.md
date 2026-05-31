# Diff summary

- **files changed:** 25
- **lines:** +51 / -61
- **kernel-ish files:** 25

## Files (by churn)

- `tile_engine/ops/gemm/gemm_common.hpp`  (+0/-52)
- `include/ck_tile/ops/gemm.hpp`  (+5/-3)
- `include/ck_tile/ops/gemm_quant.hpp`  (+5/-3)
- `include/ck_tile/ops/common.hpp`  (+2/-1)
- `include/ck_tile/ops/add_rmsnorm2d_rdquant.hpp`  (+2/-0)
- `include/ck_tile/ops/batched_transpose.hpp`  (+2/-0)
- `include/ck_tile/ops/common/load_interleaved_pk_type.hpp`  (+1/-1)
- `include/ck_tile/ops/elementwise.hpp`  (+2/-0)
- `include/ck_tile/ops/epilogue.hpp`  (+2/-0)
- `include/ck_tile/ops/flatmm.hpp`  (+2/-0)
- `include/ck_tile/ops/fmha.hpp`  (+2/-0)
- `include/ck_tile/ops/fused_moe.hpp`  (+2/-0)
- `include/ck_tile/ops/grouped_convolution.hpp`  (+2/-0)
- `include/ck_tile/ops/image_to_column.hpp`  (+2/-0)
- `include/ck_tile/ops/layernorm2d.hpp`  (+2/-0)

## Key added lines (kernel files)

**`tile_engine/ops/gemm/gemm_profiler.hpp`**
```
ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
```
