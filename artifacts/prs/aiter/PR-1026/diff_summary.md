# Diff summary

- **files changed:** 5 (diff was byte-capped; summary is partial)
- **lines:** +558 / -508
- **kernel-ish files:** 5

## Files (by churn)

- `hsa/gfx942/fmha_v3_bwd/codegen.py`  (+405/-408)
- `csrc/py_itfs_cu/asm_mha_bwd.cu`  (+71/-71)
- `aiter/ops/mha.py`  (+46/-18)
- `csrc/py_itfs_cu/asm_mha_varlen_bwd.cu`  (+28/-9)
- `csrc/include/mha_bwd.h`  (+8/-2)

## Key added lines (kernel files)

**`aiter/ops/mha.py`**
```
ret = hdim_q == 64 and is_v3_atomic_fp32 == True
ret &= not deterministic
(_, seqlen_q, nhead_q, hdim_q) = q.shape
(_, seqlen_k, nhead_k, hdim_v) = v.shape
```

**`csrc/include/mha_bwd.h`**
```
const void *ptr_qseq;
const void *ptr_qseq_padded;
unsigned int max_seqlen_dq;
bool kIsGroupMode_,
```

**`csrc/py_itfs_cu/asm_mha_bwd.cu`**
```
k.data_ptr(),
v.data_ptr(),
alibi_slopes_ptr, // bias
out.data_ptr(),
```

**`csrc/py_itfs_cu/asm_mha_varlen_bwd.cu`**
```
std::pair<uint64_t*, uint64_t*> drop_seed_offset,
bool is_v3_atomic_fp32)
ck_tile::index_t split_stride_dq_acc;
ck_tile::index_t batch_stride_dq_acc;
```

**`hsa/gfx942/fmha_v3_bwd/codegen.py`**
```
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,    false,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      1,    false,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      2,    false,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,    false,  
```
