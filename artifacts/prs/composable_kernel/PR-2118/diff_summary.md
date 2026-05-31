# Diff summary

- **files changed:** 2 (diff was byte-capped; summary is partial)
- **lines:** +168 / -1751
- **kernel-ish files:** 2

## Files (by churn)

- `example/ck_tile/01_fmha/hsaco/bwd_bf16_a32_rtna_pssk_group.cpp`  (+0/-1583)
- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+168/-168)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      0,    false,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      1,    false,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      2,    false,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      0,    false,  
```
