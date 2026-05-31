# Diff summary

- **files changed:** 24
- **lines:** +85 / -198
- **kernel-ish files:** 17

## Files (by churn)

- `csrc/py_itfs_cu/fmha_v3_bwd_kernel_generate.py`  (+38/-175)
- `op_tests/cpp/mha/smoke_test_bwd_v3.sh`  (+24/-0)
- `aiter/dist/parallel_state.py`  (+4/-4)
- `aiter/dist/custom_all_reduce.py`  (+3/-3)
- `aiter/dist/communication_op.py`  (+2/-2)
- `csrc/include/custom_all_reduce.h`  (+2/-2)
- `aiter/ops/custom_all_reduce.py`  (+1/-1)
- `csrc/include/activation.h`  (+1/-1)
- `csrc/include/cache.h`  (+1/-1)
- `csrc/include/custom_all_reduce.cuh`  (+1/-1)
- `csrc/include/moe_op.h`  (+1/-1)
- `csrc/kernels/activation_kernels.cu`  (+1/-1)
- `csrc/kernels/cache_kernels.cu`  (+1/-1)
- `csrc/kernels/custom_all_reduce.cu`  (+1/-1)
- `csrc/kernels/moe_align_block_size_kernels.cu`  (+1/-1)

## Key added lines (kernel files)

**`aiter/dist/custom_all_reduce.py`**
```
The main responsibility of this context manager is the
```

**`aiter/dist/parallel_state.py`**
```
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to
```

**`csrc/py_itfs_cu/fmha_v3_bwd_kernel_generate.py`**
```
from jit.utils.chip_info import get_gfx
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,    false,   f
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,    false,   f
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,    false,   f
```
