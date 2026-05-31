# Diff summary

- **files changed:** 7 (diff was byte-capped; summary is partial)
- **lines:** +462 / -467
- **kernel-ish files:** 7

## Files (by churn)

- `hsa/gfx942/fmha_v3_bwd/codegen.py`  (+442/-443)
- `csrc/py_itfs_cu/asm_mha_bwd.cu`  (+4/-7)
- `csrc/py_itfs_ck/mha_bwd_kernels.cu`  (+4/-4)
- `csrc/py_itfs_cu/asm_mha_varlen_bwd.cu`  (+2/-6)
- `aiter/ops/mha.py`  (+4/-3)
- `csrc/include/mha_bwd.h`  (+4/-2)
- `csrc/py_itfs_ck/mha_varlen_bwd_kernels.cu`  (+2/-2)

## Key added lines (kernel files)

**`aiter/ops/mha.py`**
```
(hdim_q > 64 and hdim_q <= 128)
or (hdim_q == 192 and hdim_v == 128 and nmask)
) and hdim_q % 8 == 0
```

**`csrc/include/mha_bwd.h`**
```
template <ck_tile::index_t HDim_q_,
ck_tile::index_t HDim_v_,
static constexpr ck_tile::index_t HDim_q  = HDim_q_;
static constexpr ck_tile::index_t HDim_v  = HDim_v_;
```

**`csrc/py_itfs_ck/mha_bwd_kernels.cu`**
```
dq_accum = torch::zeros({1, batch_size, seqlen_q, num_heads, head_size_q}, opts.dtype(at::kFloat));
const ck_tile::index_t kN0 = head_size_q <= 128 ? 128 : 64;
dq_accum = torch::empty({nsplits, batch_size, seqlen_q, num_heads, head_size_q}, opts.dtype(at::kFloat));
dq_accum = torch::zeros({nsplits, batch_size, seqlen_q, num_heads, head_size_q}, opts.dtype(at::kFloat));
```

**`csrc/py_itfs_ck/mha_varlen_bwd_kernels.cu`**
```
dq_accum = torch::zeros({1, total_q, num_heads, head_size_q}, opts.dtype(at::kFloat));
dq_accum = torch::zeros({nsplits, total_q, num_heads, head_size_q}, opts.dtype(at::kFloat));
```

**`csrc/py_itfs_cu/asm_mha_bwd.cu`**
```
dq_accum = torch::zeros({1, batch_size, num_heads, seqlen_q, head_size_q}, opts.dtype(at::kFloat));
int padded_head_size_q = head_size_q == 192? 192: 128;
dq_accum = torch::zeros({1, batch_size, num_heads, (seqlen_q + 15) / 16 * 16, padded_head_size_q}, opts.dtype(q_dtype));
```
