# Diff summary

- **files changed:** 2 (diff was byte-capped; summary is partial)
- **lines:** +2500 / -17
- **kernel-ish files:** 2

## Files (by churn)

- `example/ck_tile/01_fmha/hsaco/bwd_bf16_a32_rtna_pssk_group.cpp`  (+2276/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+224/-17)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
unsigned int head_dim;
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      0,     true,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      1,     true,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      2,     true,  
```

**`example/ck_tile/01_fmha/hsaco/bwd_bf16_a32_rtna_pssk_group.cpp`**
```
unsigned char bwd_bf16_a32_rtna_pssk_group[] = {
0x7F, 0x45, 0x4C, 0x46, 0x02, 0x01, 0x01, 0x40, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x03, 0x00, 0xE0, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0x8B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
```
