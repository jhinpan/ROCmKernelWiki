# Diff summary

- **files changed:** 18
- **lines:** +712 / -521
- **kernel-ish files:** 6

## Files (by churn)

- `hsa/gfx950/fmha_v3_bwd/codegen.py`  (+628/-472)
- `op_tests/cpp/mha/benchmark_mha_bwd.cpp`  (+22/-14)
- `op_tests/cpp/mha/smoke_test_bwd_v3.sh`  (+14/-14)
- `csrc/include/mha_bwd.h`  (+19/-6)
- `hsa/gfx942/fmha_v3_bwd/codegen.py`  (+15/-8)
- `aiter/ops/mha.py`  (+6/-5)
- `csrc/py_itfs_cu/asm_mha_bwd.cu`  (+7/-1)
- `aiter/jit/optCompilerConfig.json`  (+1/-1)
- `hsa/gfx950/fmha_v3_bwd/bwd_dq_shuffle.co`  (+0/-0)
- `hsa/gfx950/fmha_v3_bwd/bwd_hd128_bf16_a16_pssk.co`  (+0/-0)
- `hsa/gfx950/fmha_v3_bwd/bwd_hd128_bf16_a32_pssk.co`  (+0/-0)
- `hsa/gfx950/fmha_v3_bwd/bwd_hd128_bf16_causal_a16_pssk.co`  (+0/-0)
- `hsa/gfx950/fmha_v3_bwd/bwd_hd128_bf16_causal_a32_pssk.co`  (+0/-0)
- `hsa/gfx950/fmha_v3_bwd/bwd_hd128_fp16_a16_pssk.co`  (+0/-0)
- `hsa/gfx950/fmha_v3_bwd/bwd_hd128_fp16_a32_pssk.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/mha.py`**
```
gfx = get_gfx()
ret = (hdim_q == 64 and gfx == "gfx942" and is_v3_atomic_fp32 == True) or (
hdim_q == 128 and gfx == "gfx950"
how_v3_bf16_cvt = 0
```

**`csrc/include/mha_bwd.h`**
```
void* ptr_dq_acc;
void* ptr_dq;
unsigned int Hs_dq_acc;
unsigned int BAs_dq_acc;
```

**`csrc/py_itfs_cu/asm_mha_bwd.cu`**
```
if (is_v3_atomic_fp32) {
dq_accum = torch::zeros({1, batch_size, num_heads, seqlen_q, head_size_v}, opts.dtype(at::kFloat));
dq_accum = torch::zeros({1, batch_size, num_heads, (seqlen_q + 15) / 16 * 16, 128}, opts.dtype(q_dtype));
```

**`hsa/gfx942/fmha_v3_bwd/codegen.py`**
```
int gdx = (fmha_v3_traits.sk + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
int num_tg = (fmha_v3_traits.sk + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
int gdx = (fmha_v3_traits.sk + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
int num_tg = (fmha_v3_traits.sk + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
```

**`hsa/gfx950/fmha_v3_bwd/codegen.py`**
```
struct __attribute__((packed)) fmha_bwd_v3_args_gfx950
void *ptr_dq; //dq or dq_acc 0x0
void *ptr_dk;   // 0x10
void *ptr_dv;   // 0x20
```
