# Diff summary

- **files changed:** 19
- **lines:** +239 / -482
- **kernel-ish files:** 11

## Files (by churn)

- `op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py`  (+63/-95)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py`  (+78/-43)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py`  (+6/-66)
- `op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py`  (+13/-55)
- `op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py`  (+7/-53)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16.json`  (+24/-24)
- `op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py`  (+5/-43)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py`  (+7/-41)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py`  (+9/-39)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8WFP4.json`  (+5/-5)
- `aiter/ops/triton/_triton_kernels/activation.py`  (+4/-4)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-gated.json`  (+3/-3)
- `aiter/ops/triton/gluon/gemm_a8w8.py`  (+4/-2)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-N=128-K=2880.json`  (+2/-2)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-N=2880-K=4096.json`  (+2/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/activation.py`**
```
def _silu_exp2(x):
return x / (1.0 + tl.exp2(-(x * 1.44269504089)))
def _silu(x):
return _silu_exp2(x)
```

**`aiter/ops/triton/gluon/gemm_a8w8.py`**
```
mfma_instr_k: gl.constexpr = 128 if FP8_FORMAT is not None else 64
instr_shape=[16, 16, mfma_instr_k],
mfma_instr_k: gl.constexpr = 128 if FP8_FORMAT is not None else 64
instr_shape=[16, 16, mfma_instr_k],
```

**`op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py`**
```
x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
x_vals += [(2**i, 256, 7168) for i in range(5, 9)]  # DSR1 router GEMM
x_vals += [(2**i, 5120, 2880) for i in range(5, 9)]  # GPTOSS QKV input projection
x_vals += [(2**i, 2880, 4096) for i in range(5, 9)]  # output projection
```

**`op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py`**
```
x_vals = [(1, 1, 1)]  # minimal case
x_vals += [(3, 5, 2)]  # irregular shape
x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
x_vals += [(2**i, 256, 7168) for i in range(5, 9)]  # DSR1 router GEMM
```

**`op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py`**
```
x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
x_vals += [(v, 128, 512) for v in (128, 192, 4096, 8000)]
x_vals += [(v, 2112, 7168) for v in (128, 192, 4096, 8000)]
dtype = torch.bfloat16
```
