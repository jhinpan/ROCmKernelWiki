# Diff summary

- **files changed:** 10
- **lines:** +205 / -216
- **kernel-ish files:** 8

## Files (by churn)

- `scripts/run_tests.sh`  (+83/-146)
- `kernels/mfma_preshuffle_pipeline.py`  (+34/-18)
- `tests/kernels/test_moe_gemm.py`  (+26/-21)
- `tests/kernels/test_gpu_with_rocir_coords.py`  (+10/-13)
- `kernels/moe_gemm_2stage.py`  (+16/-5)
- `kernels/preshuffle_gemm.py`  (+15/-4)
- `tests/kernels/test_preshuffle_gemm.py`  (+11/-7)
- `tests/conftest.py`  (+8/-0)
- `.github/workflows/flydsl.yaml`  (+1/-1)
- `tests/kernels/test_vec_add.py`  (+1/-1)

## Key added lines (kernel files)

**`kernels/mfma_preshuffle_pipeline.py`**
```
elem_bytes: int = 1,
"""Load 16 bytes from global memory via RawPtrBufferLoadOp (dwordx4).
Always uses buffer instruction for all element types, ensuring hardware OOB
protection via the buffer resource's `num_records`.
```

**`kernels/moe_gemm_2stage.py`**
```
elem_bytes=elem_bytes,
a_kpack_elems = 16 // elem_bytes
col_offset_base = lane_div_16 * arith.constant(int(a_kpack_elems), index=True)
elem_bytes=elem_bytes,
```

**`kernels/preshuffle_gemm.py`**
```
_i32 = T.i32
c_m_i32 = arith.index_cast(_i32, c_m)
c_n_i32 = arith.index_cast(_i32, c_n)
c_k_i32 = arith.index_cast(_i32, c_k)
```

**`tests/conftest.py`**
```
def pytest_configure(config):
"""Register custom markers."""
config.addinivalue_line(
"markers",
```

**`tests/kernels/test_gpu_with_rocir_coords.py`**
```
assert(max_error == 0, "Max absolute difference is not 0")
test_matmul_with_flir()
print(" TEST PASSED!")
print("\nDemonstrated:")
```
