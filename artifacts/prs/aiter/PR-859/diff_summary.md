# Diff summary

- **files changed:** 73
- **lines:** +231 / -182
- **kernel-ish files:** 8

## Files (by churn)

- `hsa/gfx942/fmha_v3_bwd/codegen.py`  (+73/-64)
- `hsa/gfx950/fmha_v3_bwd/codegen.py`  (+73/-64)
- `csrc/include/moe_op.h`  (+37/-37)
- `csrc/py_itfs_cu/asm_mha_varlen_bwd.cu`  (+12/-8)
- `csrc/include/mha_bwd.h`  (+16/-3)
- `csrc/cpp_itfs/mha_bwd_generate.py`  (+6/-4)
- `aiter/ops/mha.py`  (+8/-0)
- `csrc/include/torch/mha_v3_varlen_bwd.h`  (+5/-1)
- `op_tests/cpp/mha/smoke_test_bwd_v3.sh`  (+1/-1)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtna_pssk_group.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtna_psskddv_group.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtne_pssk_group.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtne_psskddv_group.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtz_pssk_group.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtz_psskddv_group.co`  (+0/-0)

## Key added lines (kernel files)

**`csrc/cpp_itfs/mha_bwd_generate.py`**
```
int how_v3_bf16_cvt,
const void* seqlen_q_padded,
const void* seqlen_k_padded)
t = gfx942::fmha_bwd_v3(traits, args, stream_config, seqlen_q_padded, seqlen_k_padded);
```

**`csrc/include/mha_bwd.h`**
```
int how_v3_bf16_cvt,
const void* seqlen_q_padded = nullptr,
const void* seqlen_k_padded = nullptr);
const void* ptr_qseq_padded;
```

**`csrc/include/moe_op.h`**
```
bool scoring_func                 = true,
torch::Tensor& out,               // [token_cnt, dim]
torch::Tensor& input,             // [token_cnt, dim] M,K
torch::Tensor& gate,              // [expert, hidden_dim*2, dim] N,K
```

**`csrc/py_itfs_cu/asm_mha_varlen_bwd.cu`**
```
fmha_v3_varlen_bwd(const at::Tensor &dout,                  // [total_q, hq, d_v]
const at::Tensor &q,                     // [total_q, hq, d_q]
const at::Tensor &k,                     // [total_k, hk, d_q]
const at::Tensor &v,                     // [total_k, hk, d_v]
```

**`hsa/gfx942/fmha_v3_bwd/codegen.py`**
```
float fmha_bwd_v3_group_(const ck_tile::stream_config& s, fmha_bwd_args a, const void* seqlen_q_padded = nullptr, const 
args.ptr_dq             = a.dq_acc_ptr;
args.ptr_dk             = a.dk_ptr;
args.ptr_dv             = a.dv_ptr;
```
