# Diff summary

- **files changed:** 20
- **lines:** +834 / -214
- **kernel-ish files:** 20

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/pa_decode.py`  (+156/-29)
- `aiter/ops/triton/hstu_attention.py`  (+41/-68)
- `aiter/ops/triton/lean_atten.py`  (+86/-6)
- `aiter/ops/triton/pod_attention.py`  (+61/-5)
- `aiter/ops/triton/_triton_kernels/hstu_attention.py`  (+38/-24)
- `aiter/ops/triton/_triton_kernels/mha_fused_bwd.py`  (+50/-8)
- `aiter/ops/triton/_triton_kernels/mha_onekernel_bwd.py`  (+53/-5)
- `aiter/ops/triton/mla_decode_rope.py`  (+28/-23)
- `aiter/ops/triton/_triton_kernels/mla_decode_rope.py`  (+39/-6)
- `aiter/ops/triton/mha_fused_bwd.py`  (+42/-3)
- `aiter/ops/triton/mha_onekernel_bwd.py`  (+42/-2)
- `aiter/ops/triton/pa_decode.py`  (+21/-18)
- `aiter/ops/triton/_triton_kernels/pod_attention.py`  (+31/-2)
- `aiter/ops/triton/_triton_kernels/lean_atten.py`  (+27/-4)
- `aiter/ops/triton/_triton_kernels/pa_prefill.py`  (+28/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/chunked_pa_prefill.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_kernel_paged_attention_2d_repr = make_kernel_repr(
"_kernel_paged_attention_2d",
"num_queries_per_kv",
```

**`aiter/ops/triton/_triton_kernels/hstu_attention.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_hstu_attn_fwd_repr = make_kernel_repr(
"_hstu_attn_fwd",
"CAUSAL",
```

**`aiter/ops/triton/_triton_kernels/lean_atten.py`**
```
from ..utils._triton.pid_preprocessing import remap_xcd
from ..utils._triton.kernel_repr import make_kernel_repr
_la_persistent_repr = make_kernel_repr(
"la_persistent",
```

**`aiter/ops/triton/_triton_kernels/mha.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_attn_fwd_repr = make_kernel_repr(
"_attn_fwd",
"IS_CAUSAL",
```

**`aiter/ops/triton/_triton_kernels/mha_fused_bwd.py`**
```
from ..utils._triton.pid_preprocessing import remap_xcd
from ..utils._triton.kernel_repr import make_kernel_repr
_bwd_preprocess_repr = make_kernel_repr(
"_bwd_preprocess",
```
