# Diff summary

- **files changed:** 39
- **lines:** +153 / -131
- **kernel-ish files:** 39

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/chunk_delta_h.py`  (+20/-20)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/chunk_o.py`  (+18/-18)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py`  (+11/-11)
- `aiter/ops/triton/_triton_kernels/attention/mha_onekernel_bwd.py`  (+10/-8)
- `aiter/ops/triton/_triton_kernels/gather_kv_b_proj.py`  (+8/-8)
- `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_a16w16.py`  (+8/-4)
- `aiter/ops/triton/_triton_kernels/gmm.py`  (+5/-5)
- `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`  (+4/-4)
- `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py`  (+4/-4)
- `aiter/ops/triton/_triton_kernels/attention/pa_decode.py`  (+4/-4)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py`  (+6/-2)
- `aiter/ops/triton/_triton_kernels/attention/mha.py`  (+3/-3)
- `aiter/ops/triton/_triton_kernels/attention/mha_fused_bwd.py`  (+3/-3)
- `aiter/ops/triton/_triton_kernels/attention/extend_attention.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/attention/hstu_attention.py`  (+2/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/attention/extend_attention.py`**
```
qk = tl.dot(qpe.to(kpe.dtype), kpe, acc=qk)
qk = tl.dot(qpe, kpe, acc=qk)
```

**`aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`**
```
acc = tl.dot((p).to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)
acc = tl.dot((p).to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)
acc = tl.dot((p).to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)
acc = tl.dot((p).to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)
```

**`aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py`**
```
acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)
acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)
acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)
acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)
```

**`aiter/ops/triton/_triton_kernels/attention/hstu_attention.py`**
```
dv = tl.dot(silu_trans, do, allow_tf32=ALLOW_TF32, acc=dv)
dk = tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32, acc=dk)
```

**`aiter/ops/triton/_triton_kernels/attention/lean_atten.py`**
```
qk = tl.dot(q, k, acc=qk)
acc = tl.dot(p.to(v.dtype), v, acc=acc)
```
