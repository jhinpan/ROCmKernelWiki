# Diff summary

- **files changed:** 11
- **lines:** +168 / -25
- **kernel-ish files:** 11

## Files (by churn)

- `op_tests/triton_tests/test_gemm_a16w16.py`  (+44/-0)
- `op_tests/triton_tests/test_batched_gemm_bf16.py`  (+37/-1)
- `op_tests/triton_tests/test_batched_gemm_a8w8.py`  (+35/-0)
- `op_tests/triton_tests/test_gemm_afp4wfp4.py`  (+12/-4)
- `op_tests/triton_tests/test_gemm_a8w8.py`  (+11/-4)
- `op_tests/triton_tests/test_gemm_a8w8_per_token_scale.py`  (+6/-4)
- `op_tests/triton_tests/test_gemm_a16w16_gated.py`  (+7/-2)
- `op_tests/triton_tests/test_gemm_a8w8_blockscale.py`  (+5/-3)
- `op_tests/triton_tests/test_batched_gemm_afp4wfp4_pre_quant.py`  (+5/-2)
- `op_tests/triton_tests/test_gemm_afp4wfp4_pre_quant_atomic.py`  (+3/-3)
- `op_tests/triton_tests/test_batched_gemm_afp4wfp4.py`  (+3/-2)

## Key added lines (kernel files)

**`op_tests/triton_tests/test_batched_gemm_a8w8.py`**
```
import functools
def minimal_x_vals(num_vals=20):
Returns the num_vals smallest test cases. Useful for generating a subset to quickly test on.
x_vals = get_x_vals()
```

**`op_tests/triton_tests/test_batched_gemm_afp4wfp4.py`**
```
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
def test_batched_gemm_afp4_wfp4(B: int, M: int, N: int, K: int, dtype, layout):
B, M, N, K, dtype, layout=layout, output=True
```

**`op_tests/triton_tests/test_batched_gemm_afp4wfp4_pre_quant.py`**
```
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
def test_batched_gemm_afp4_wfp4_pre_quant(
B: int, M: int, N: int, K: int, layout, dtype
B, M, N, K, dtype, layout=layout, output=True
```

**`op_tests/triton_tests/test_batched_gemm_bf16.py`**
```
import functools
def minimal_x_vals(num_vals=20):
Returns the num_vals smallest test cases. Useful for generating a subset to quickly test on.
x_vals = get_x_vals()
```

**`op_tests/triton_tests/test_gemm_a16w16.py`**
```
@pytest.mark.parametrize("M, N, K", minimal_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
@pytest.mark.parametrize("output", [True, False])
```
