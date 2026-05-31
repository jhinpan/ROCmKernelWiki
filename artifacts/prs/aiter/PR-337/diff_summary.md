# Diff summary

- **files changed:** 78
- **lines:** +636 / -250
- **kernel-ish files:** 4

## Files (by churn)

- `csrc/py_itfs_cu/fmha_bwd_pre_post_kernel.py`  (+485/-170)
- `aiter/ops/mha.py`  (+94/-60)
- `op_tests/cpp/mha/smoke_test_bwd_v3.sh`  (+57/-16)
- `test_mha.sh`  (+0/-2)
- `csrc/py_itfs_ck/mha_bwd_kernels.cu`  (+0/-1)
- `csrc/py_itfs_ck/mha_varlen_bwd_kernels.cu`  (+0/-1)
- `hsa/fmha_v3_bwd/bwd_hd128_bf16_a16_rtna.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_hd128_bf16_a16_rtna_pddv.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_hd128_bf16_a16_rtne.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_hd128_bf16_a16_rtne_pddv.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_hd128_bf16_a16_rtz.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_hd128_bf16_a16_rtz_pddv.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_hd128_bf16_a32_rtna.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_hd128_bf16_a32_rtna_pssk_group.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_hd128_bf16_a32_rtna_psskddv.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/mha.py`**
```
ret &= hdim_q > 64 and hdim_q <= 192
ret &= dbias is None
ret &= hdim_q >= 64 and hdim_q <= 192 and hdim_q % 8 == 0
ret &= hdim_q == 64 or hdim_q == 128
```

**`csrc/py_itfs_cu/fmha_bwd_pre_post_kernel.py`**
```
unsigned int nhead_q;
unsigned int Hs_q;
unsigned int BAs_q;
unsigned int Seqs_q;
```
