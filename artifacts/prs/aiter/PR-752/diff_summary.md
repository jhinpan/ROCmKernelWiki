# Diff summary

- **files changed:** 7
- **lines:** +145 / -68
- **kernel-ish files:** 7

## Files (by churn)

- `op_tests/test_moe.py`  (+103/-54)
- `op_tests/test_moe_blockscale.py`  (+25/-1)
- `op_tests/test_batched_gemm_a8w8.py`  (+7/-5)
- `op_tests/test_batched_gemm_bf16.py`  (+7/-5)
- `op_tests/test_gemm_a4w4.py`  (+1/-1)
- `op_tests/test_gemm_a8w8.py`  (+1/-1)
- `op_tests/test_kvcache_blockscale.py`  (+1/-1)

## Key added lines (kernel files)

**`op_tests/test_batched_gemm_a8w8.py`**
```
help="""Shape of mnk.
e.g.:   -s 1024,8192,1024
--mnk 1024,8192,1024""",
l_b = [args.batch]
```

**`op_tests/test_batched_gemm_bf16.py`**
```
help="""Shape of mnk.
e.g.    -s 1024,8192,1024
--mnk 1024,8192,1024""",
l_b = [args.batch]
```

**`op_tests/test_gemm_a4w4.py`**
```
help="""Shape of mnk.
```

**`op_tests/test_gemm_a8w8.py`**
```
help="""Shape of mnk.
```

**`op_tests/test_kvcache_blockscale.py`**
```
help="""Select which test to run, default is all
```
