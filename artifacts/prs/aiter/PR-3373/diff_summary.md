# Diff summary

- **files changed:** 18
- **lines:** +577 / -523
- **kernel-ish files:** 14

## Files (by churn)

- `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`  (+140/-176)
- `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`  (+104/-120)
- `aiter/ops/flydsl/moe_kernels.py`  (+73/-60)
- `aiter/ops/flydsl/kernels/qk_norm_rope_quant.py`  (+47/-30)
- `aiter/ops/flydsl/kernels/preshuffle_gemm.py`  (+39/-30)
- `aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py`  (+42/-20)
- `aiter/ops/flydsl/kernels/silu_and_mul_fq.py`  (+26/-21)
- `aiter/ops/flydsl/gemm_kernels.py`  (+22/-12)
- `aiter/ops/flydsl/kernels/small_m_hgemm.py`  (+14/-16)
- `aiter/ops/flydsl/kernels/splitk_hgemm.py`  (+14/-16)
- `aiter/aot/flydsl/gemm.py`  (+27/-2)
- `aiter/ops/flydsl/kernels/tensor_shim.py`  (+13/-6)
- `aiter/aot/flydsl/moe.py`  (+8/-7)
- `setup.py`  (+3/-2)
- `.github/workflows/atom-test.yaml`  (+2/-2)

## Key added lines (kernel files)

**`aiter/aot/flydsl/gemm.py`**
```
def _ptr_view_safe(t):
from aiter.ops.flydsl.gemm_kernels import _ptr_view_safe as _wrap
return _wrap(t)
_ptr_view_safe(out),
```

**`aiter/aot/flydsl/moe.py`**
```
_ptr_view_safe,
_ptr_view_safe(tmp_out.view(-1, inter_dim * 2)),
_ptr_view_safe(out.view(-1).view(torch.uint8)),
_ptr_view_safe(out_scale_sorted_flat),
```

**`aiter/ops/flydsl/gemm_kernels.py`**
```
import flydsl.compiler as flyc
def _ptr_view_safe(t: torch.Tensor):
type_name = type(t).__name__
module_name = type(t).__module__
```

**`aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py`**
```
def _pointer_to_llvm_ptr(ptr) -> ir.Value:
"""Convert a FlyDSL pointer argument to the LLVM pointer used by raw loads."""
ptr_i64 = arith.index_cast(T.i64, fx.ptrtoint(ptr))
return _llvm.IntToPtrOp(_llvm_ptr_ty(), ptr_i64).result
```

**`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`**
```
w_elem_bytes = 2 if is_f16_b else 1
w_elem_pack = 2 if (is_f4_b or is_int4) else 1
w_nbytes = (experts * (2 * inter_dim) * model_dim * w_elem_bytes) // w_elem_pack
bias_nbytes = experts * (2 * inter_dim) * 4
```
