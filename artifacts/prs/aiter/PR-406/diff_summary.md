# Diff summary

- **files changed:** 85
- **lines:** +98 / -97
- **kernel-ish files:** 1

## Files (by churn)

- `csrc/py_itfs_cu/fmha_bwd_pre_post_kernel.py`  (+98/-97)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a16_rtna.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a16_rtna_pddv.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a16_rtne.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a16_rtne_pddv.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a16_rtz.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a16_rtz_pddv.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtna.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtna_pssk_group.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtna_psskddv.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtna_psskddv_group.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtne.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtne_pssk_group.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtne_psskddv.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtne_psskddv_group.co`  (+0/-0)

## Key added lines (kernel files)

**`csrc/py_itfs_cu/fmha_bwd_pre_post_kernel.py`**
```
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,    false,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      1,    false,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      2,    false,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,    false,  
```
