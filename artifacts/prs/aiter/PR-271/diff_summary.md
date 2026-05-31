# Diff summary

- **files changed:** 27
- **lines:** +589 / -300
- **kernel-ish files:** 26

## Files (by churn)

- `csrc/py_itfs_ck/moe_ck_2stages_kernel.cu`  (+166/-81)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm.hpp`  (+154/-45)
- `op_tests/test_moe_2stage.py`  (+61/-39)
- `aiter/fused_moe_bf16_asm.py`  (+35/-30)
- `op_tests/test_moe_2stage_int4.py`  (+34/-22)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm_common.cuh`  (+23/-23)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_b16.cu`  (+10/-5)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_b16_f8.cu`  (+10/-5)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_f16.cu`  (+10/-5)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_f16_f8.cu`  (+10/-5)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_b16.cu`  (+10/-5)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_b16_f8.cu`  (+10/-5)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_f16.cu`  (+10/-5)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_f16_f8.cu`  (+10/-5)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_b16_f8_wint4.cu`  (+6/-3)

## Key added lines (kernel files)

**`aiter/fused_moe_bf16_asm.py`**
```
if activation == ActivationType.Silu:
act_op = 2
act_op = 0
(M, topk, w1.shape[1] // 2),
```

**`aiter/jit/core.py`**
```
"-mllvm", "--amdgpu-enable-max-ilp-scheduling-strategy=1",
```

**`aiter/ops/moe_op.py`**
```
block_m: Optional[int] = 32,
ActOP: Optional[int] = 2
```

**`csrc/include/moe_ck.h`**
```
std::optional<int> block_m,
std::optional<int> ActOP);
```

**`csrc/include/rocm_ops.hpp`**
```
py::arg("block_m") = 32,            \
py::arg("ActOP")  = 2);             \
```
