# Diff summary

- **files changed:** 17
- **lines:** +370 / -248
- **kernel-ish files:** 11

## Files (by churn)

- `docs/layout_system_guide.md`  (+89/-12)
- `kernels/preshuffle_gemm.py`  (+35/-37)
- `docs/tutorials/kernel_development.rst`  (+35/-25)
- `docs/kernel_authoring_guide.md`  (+46/-7)
- `kernels/mfma_preshuffle_pipeline.py`  (+23/-26)
- `kernels/blockscale_preshuffle_gemm.py`  (+20/-22)
- `kernels/moe_gemm_2stage.py`  (+20/-21)
- `kernels/mixed_moe_gemm_2stage.py`  (+20/-20)
- `kernels/moe_blockscale_2stage.py`  (+19/-20)
- `kernels/mixed_preshuffle_gemm.py`  (+18/-20)
- `docs/cute_layout_algebra_guide.md`  (+8/-21)
- `docs/tutorials/basic_usage.rst`  (+14/-6)
- `docs/api/dsl.rst`  (+8/-6)
- `python/flydsl/expr/primitive.py`  (+9/-0)
- `kernels/moe_reduce.py`  (+3/-3)

## Key added lines (kernel files)

**`kernels/blockscale_preshuffle_gemm.py`**
```
from flydsl.expr.arith import ArithValue
n_blk_list.append(global_n // 16)
k0_base = base_k_bytes // c64_b
buffer_ops, vector, b_rsrc, idx_pack,
```

**`kernels/kernels_common.py`**
```
from flydsl.expr.typing import T
i64_type = T.i64
```

**`kernels/mfma_epilogues.py`**
```
m_lane = tx // c_nlane
```

**`kernels/mfma_preshuffle_pipeline.py`**
```
idx_i32 = idx // 4
idx_i32 = (idx * 2) // 4
c_k0 = c_k_bytes // c64
n0 = c_n // c16
```

**`kernels/mixed_moe_gemm_2stage.py`**
```
c_k_scale = c_k // scale_block_size
c_mn1 = c_mn // c16 // c_mn_pack
c_k1 = c_k_scale // c4 // c_k_pack
inter_in = arith.ArithValue(arith.index_cast(ir.IndexType.get(), i32_inter_in.ir_value()))
```
