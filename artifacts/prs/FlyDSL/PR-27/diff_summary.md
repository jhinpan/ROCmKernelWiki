# Diff summary

- **files changed:** 6
- **lines:** +627 / -359
- **kernel-ish files:** 6

## Files (by churn)

- `tests/python/gpu/test_moe_gemm.py`  (+365/-147)
- `tests/python/gpu/test_preshuffle_gemm.py`  (+109/-151)
- `tests/python/gpu/test_gemm.py`  (+68/-45)
- `tests/python/gpu/mfma_fp8_preshuffle_pipeline.py`  (+64/-11)
- `pyflir/src/pyflir/dialects/ext/flir.py`  (+18/-5)
- `pyflir/src/pyflir/lang/ir/types.py`  (+3/-0)

## Key added lines (kernel files)

**`pyflir/src/pyflir/dialects/ext/flir.py`**
```
def emit_tensor_load(copy_shape, src_view: TensorView, pred_val: Optional[Value] = None):
mask = _unwrap_value(pred_val) if pred_val is not None else None
mask=mask,
vector_dialect.store(
```

**`tests/python/gpu/mfma_fp8_preshuffle_pipeline.py`**
```
"""CK-style XOR16 swizzle on K at 16B granularity (index-typed).
This now routes through the dedicated `flir.swizzle_xor16` op so lowering can
optimize to bitwise ops when `kBlocks16` is a const power-of-two.
return _unwrap(flir.swizzle_xor16(row_idx, col_idx, k_blocks16))
```

**`tests/python/gpu/test_gemm.py`**
```
c0_i32 = arith.i32(0).value
linear_id = tx * vec_len_val
coord_a_base = flir.make_coord(row_a_global, col_a_local)
return _arith_mlir.XOrIOp(
```

**`tests/python/gpu/test_moe_gemm.py`**
```
import argparse
from pyflir.lang.ir.types import T as I
make_preshuffle_b_layout,
(256, 4096, 2048, 17, 9, 64, 128, 128, False),
```

**`tests/python/gpu/test_preshuffle_gemm.py`**
```
RUN_AITER_BENCH = os.environ.get("COMPARE_AITER_CK", "1") == "1"
a_load_bytes = 16
if bytes_per_thread_a % a_load_bytes != 0:
raise ValueError(
```
