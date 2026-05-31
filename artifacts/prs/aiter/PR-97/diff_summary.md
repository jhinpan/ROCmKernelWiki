# Diff summary

- **files changed:** 24
- **lines:** +100 / -30
- **kernel-ish files:** 14

## Files (by churn)

- `csrc/py_itfs_cu/asm_pa.cpp`  (+31/-11)
- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+24/-13)
- `csrc/py_itfs_ck/norm_kernels.cu`  (+7/-0)
- `csrc/py_itfs_ck/rmsnorm_ck_kernels.cu`  (+7/-0)
- `csrc/py_itfs_cu/custom.cu`  (+5/-1)
- `csrc/ck_gemm_a8w8/include/gemm_a8w8_common.cuh`  (+4/-1)
- `csrc/include/aiter_hip_common.h`  (+5/-0)
- `op_tests/test_pa.py`  (+3/-2)
- `op_tests/test_moe.py`  (+2/-2)
- `csrc/py_itfs_ck/smoothquant_kernels.cu`  (+3/-0)
- `csrc/py_itfs_cu/asm_layernorm.cpp`  (+3/-0)
- `csrc/py_itfs_ck/moe_kernels.cu`  (+2/-0)
- `csrc/py_itfs_ck/moe_sorting_kernels.cu`  (+2/-0)
- `csrc/py_itfs_cu/asm_gemm_a8w8.cpp`  (+2/-0)
- `hsa/fmoe_b16.co`  (+0/-0)

## Key added lines (kernel files)

**`csrc/ck_gemm_a8w8/include/gemm_a8w8_common.cuh`**
```
const at::cuda::OptionalCUDAGuard device_guard(device_of(XQ));
const at::cuda::OptionalCUDAGuard device_guard(device_of(XQ));
```

**`csrc/include/aiter_hip_common.h`**
```
~AiterAsmKernel()
HIP_CALL(hipModuleUnload(module));
```

**`csrc/py_itfs_ck/moe_kernels.cu`**
```
const at::cuda::OptionalCUDAGuard device_guard(device_of(hidden_states));
```

**`csrc/py_itfs_ck/moe_sorting_kernels.cu`**
```
const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_ids));
```

**`csrc/py_itfs_ck/norm_kernels.cu`**
```
const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
```
