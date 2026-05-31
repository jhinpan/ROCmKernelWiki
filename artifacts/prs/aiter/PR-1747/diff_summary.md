# Diff summary

- **files changed:** 14
- **lines:** +470 / -6
- **kernel-ish files:** 6

## Files (by churn)

- `csrc/py_itfs_cu/asm_fmoe.cu`  (+296/-1)
- `aiter/ops/moe_op.py`  (+48/-0)
- `csrc/include/rocm_ops.hpp`  (+37/-1)
- `aiter/fused_moe.py`  (+37/-0)
- `csrc/include/moe_op.h`  (+32/-0)
- `op_tests/test_moe_2stage.py`  (+20/-4)
- `hsa/gfx942/fmoe_2stages/fmoe_small_stage1_bf16_pertokenBf16_g1u1_16x32_batch.co`  (+0/-0)
- `hsa/gfx942/fmoe_2stages/fmoe_small_stage1_bf16_pertokenBf16_g1u1_16x32_batch1.co`  (+0/-0)
- `hsa/gfx942/fmoe_2stages/fmoe_small_stage1_bf16_pertokenFp8_g1u1_16x32_batch.co`  (+0/-0)
- `hsa/gfx942/fmoe_2stages/fmoe_small_stage1_bf16_pertokenFp8_g1u1_16x32_batch1.co`  (+0/-0)
- `hsa/gfx942/fmoe_2stages/fmoe_small_stage2_bf16_pertokenBf16_g1u1_16x32_batch.co`  (+0/-0)
- `hsa/gfx942/fmoe_2stages/fmoe_small_stage2_bf16_pertokenBf16_g1u1_16x32_batch1.co`  (+0/-0)
- `hsa/gfx942/fmoe_2stages/fmoe_small_stage2_bf16_pertokenFp8_g1u1_16x32_batch.co`  (+0/-0)
- `hsa/gfx942/fmoe_2stages/fmoe_small_stage2_bf16_pertokenFp8_g1u1_16x32_batch1.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
if os.environ.get('AITER_MOE_SMALL_BATCH', '0') == '1' and hidden_states.shape[0] <= 16 and hidden_states.dtype == torch
((quant_type == QuantType.No and w1.dtype == torch.bfloat16) or (quant_type == QuantType.per_Token and w1.dtype == torch
B = hidden_states.shape[0]
E, N1, K1 = w1.shape
```

**`aiter/ops/moe_op.py`**
```
@compile_ops("module_moe_asm")
def moe_stage1_g1u1_small_batch1(
hidden_states: Tensor,
w1: Tensor,
```

**`csrc/include/moe_op.h`**
```
void moe_stage1_g1u1_small_batch1(torch::Tensor& hidden_states,             // [token_cnt, dim] M,K
torch::Tensor& w1,                        // [expert, inter_dim*2, dim] N,K
torch::Tensor& gemm1_out,                 // [token_cnt, dim]
torch::Tensor& topk_ids,                  // [token_cnt, topk]
```

**`csrc/include/rocm_ops.hpp`**
```
m.def("moe_sum", &aiter::moe_sum, "moe_sum(Tensor! input, Tensor output) -> ()");     \
m.def("moe_stage1_g1u1_small_batch1",                                      \
&moe_stage1_g1u1_small_batch1,                                       \
py::arg("hidden_states"),                                            \
```

**`csrc/py_itfs_cu/asm_fmoe.cu`**
```
protected:
class FMoeSmallBatchKernel : public FMoeKernel
struct __attribute__((packed)) Batch1KernelArgs
void* p_hidden_states;
```
