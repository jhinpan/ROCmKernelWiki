# Diff summary

- **files changed:** 15
- **lines:** +2422 / -1776
- **kernel-ish files:** 14

## Files (by churn)

- `csrc/kernels/mla/metadata.cu`  (+99/-1556)
- `csrc/kernels/mla/metadata/v1_1_device.cuh`  (+651/-0)
- `csrc/kernels/mla/metadata/v1_2_device.cuh`  (+646/-0)
- `csrc/kernels/mla/reduce.cu`  (+314/-158)
- `csrc/kernels/mla/metadata/v1_1_host.cuh`  (+264/-0)
- `csrc/kernels/mla/metadata/v1_comm.cuh`  (+218/-0)
- `csrc/kernels/mla/metadata/v0.cuh`  (+98/-0)
- `op_tests/test_mla_fp8.py`  (+40/-13)
- `op_tests/test_mla.py`  (+38/-12)
- `csrc/include/mla.h`  (+15/-32)
- `csrc/include/rocm_ops.hpp`  (+15/-2)
- `aiter/jit/core.py`  (+11/-1)
- `aiter/jit/optCompilerConfig.json`  (+6/-1)
- `aiter/mla.py`  (+6/-1)
- `aiter/ops/attention.py`  (+1/-0)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
elif origin is dict:
if not isinstance(arg, dict):
raise TypeError(
f"{el} needs to be Dict[{sub_t}] but got {arg}"
```

**`aiter/mla.py`**
```
return_lse=False,
if return_lse:
final_lse = torch.empty((total_s, nhead), dtype=dtypes.fp32, device=device)
final_lse = None
```

**`aiter/ops/attention.py`**
```
split_params: Optional[dict[str, int]] = None,
```

**`csrc/include/mla.h`**
```
int32_t batch_idx;
int32_t qo_start;
int32_t qo_end;
void get_mla_metadata_v1(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
```

**`csrc/include/rocm_ops.hpp`**
```
m.def("get_mla_metadata_v1",                        \
&get_mla_metadata_v1,                         \
"get_mla_metadata_v1",                        \
py::arg("seqlens_qo_indptr"),                 \
```
