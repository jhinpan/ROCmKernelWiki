# Diff summary

- **files changed:** 20
- **lines:** +363 / -82
- **kernel-ish files:** 11

## Files (by churn)

- `csrc/py_itfs_cu/asm_mha_varlen_fwd.cu`  (+91/-16)
- `csrc/py_itfs_cu/asm_mha_fwd.cu`  (+90/-15)
- `csrc/cpp_itfs/mha_fwd.cu`  (+49/-23)
- `aiter/ops/mha.py`  (+56/-5)
- `csrc/include/rocm_ops.hpp`  (+23/-17)
- `csrc/include/mha_fwd.h`  (+18/-0)
- `op_tests/test_mha_fp8.py`  (+12/-2)
- `op_tests/test_mha_varlen_fp8.py`  (+12/-2)
- `hsa/gfx942/fmha_v3_fwd/fmha_fwd.csv`  (+5/-1)
- `csrc/include/torch/mha_v3_fwd.h`  (+3/-0)
- `csrc/include/torch/mha_v3_varlen_fwd.h`  (+3/-0)
- `hsa/codegen.py`  (+1/-1)
- `hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_fp8.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_fp8_causal.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_fp8_causal_group.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/mha.py`**
```
q_descale: Optional[Tensor] = None,
k_descale: Optional[Tensor] = None,
v_descale: Optional[Tensor] = None,
q_descale: Optional[Tensor] = None,
```

**`csrc/cpp_itfs/mha_fwd.cu`**
```
args.ptr_o            = a.o_ptr;
args.ptr_q            = a.q_ptr;
args.ptr_k            = a.k_ptr;
args.ptr_v            = a.v_ptr;
```

**`csrc/include/mha_fwd.h`**
```
const void* ptr_q_descale;
const void* ptr_k_descale;
const void* ptr_v_descale;
unsigned int s_descale_q_Bs;
```

**`csrc/include/rocm_ops.hpp`**
```
m.def("fmha_v3_fwd",                           \
&aiter::torch_itfs::fmha_v3_fwd,         \
py::arg("q"),                            \
py::arg("k"),                            \
```

**`csrc/include/torch/mha_v3_fwd.h`**
```
std::optional<const at::Tensor> q_descale,    // [1] or [b, h_k]
std::optional<const at::Tensor> k_descale,    // [1] or [b, h_k]
std::optional<const at::Tensor> v_descale,    // [1] or [b, h_k]
```
