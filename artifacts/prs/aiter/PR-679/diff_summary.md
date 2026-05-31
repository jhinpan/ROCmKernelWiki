# Diff summary

- **files changed:** 96
- **lines:** +137 / -29
- **kernel-ish files:** 2

## Files (by churn)

- `csrc/py_itfs_cu/fmha_v3_bwd_kernel_generate.py`  (+94/-6)
- `op_tests/cpp/mha/smoke_test_bwd_v3.sh`  (+40/-20)
- `op_tests/cpp/mha/README.md`  (+2/-2)
- `aiter/ops/mha.py`  (+1/-1)
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

## Key added lines (kernel files)

**`aiter/ops/mha.py`**
```
ret &= hdim_q >= 64 and hdim_q <= 192
```

**`csrc/py_itfs_cu/fmha_v3_bwd_kernel_generate.py`**
```
unsigned int Hs_lsed;
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      0,     true,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      1,     true,  
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      2,     true,  
```
