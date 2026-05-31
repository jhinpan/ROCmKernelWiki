# Diff summary

- **files changed:** 29
- **lines:** +372 / -381
- **kernel-ish files:** 28

## Files (by churn)

- `csrc/include/aiter_tensor.h`  (+150/-2)
- `csrc/include/rocm_ops.hpp`  (+66/-62)
- `csrc/py_itfs_cu/asm_mi350_a8w8_blockscale.cu`  (+0/-117)
- `csrc/kernels/custom_all_reduce.cu`  (+31/-35)
- `aiter/ops/custom_all_reduce.py`  (+29/-27)
- `csrc/include/aiter_stream.h`  (+35/-0)
- `aiter/jit/optCompilerConfig.json`  (+3/-27)
- `csrc/pybind/aiter_enum_pybind.cu`  (+0/-26)
- `aiter/ops/gemm_op_a8w8.py`  (+0/-25)
- `csrc/include/custom_all_reduce.h`  (+7/-13)
- `aiter/utility/dtypes.py`  (+16/-2)
- `aiter/dist/device_communicators/custom_all_reduce.py`  (+1/-16)
- `csrc/include/aiter_hip_common.h`  (+7/-8)
- `aiter/jit/core.py`  (+9/-2)
- `csrc/pybind/aiter_core_pybind.cu`  (+3/-4)

## Key added lines (kernel files)

**`aiter/dist/device_communicators/custom_all_reduce.py`**
```
self._raw_ptr = ops.allocate_meta_buffer(size)
```

**`aiter/jit/core.py`**
```
rebuilded_list = ["module_aiter_core"]
develop: bool = False,
"module_aiter_core"
if develop:
```

**`aiter/ops/custom_all_reduce.py`**
```
@compile_ops("module_custom_all_reduce", develop=True)
@compile_ops("module_custom_all_reduce", develop=True)
@compile_ops("module_custom_all_reduce", develop=True)
@compile_ops("module_custom_all_reduce", develop=True)
```

**`aiter/ops/enum.py`**
```
@compile_ops("module_aiter_core", "ActivationType")
@compile_ops("module_aiter_core", "QuantType")
```

**`aiter/utility/dtypes.py`**
```
@compile_ops("module_aiter_core", "make_aiter_tensor")
def _sync_hip_stream():
"""Sync the aiter thread-local HIP stream with torch's current stream.
Called automatically by torch_to_aiter_pybind() so that C++ kernels
```
