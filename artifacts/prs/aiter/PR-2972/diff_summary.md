# Diff summary

- **files changed:** 14
- **lines:** +1080 / -167
- **kernel-ish files:** 12

## Files (by churn)

- `csrc/kernels/activation_kernels.cu`  (+502/-0)
- `aiter/fused_moe.py`  (+242/-102)
- `aiter/ops/flydsl/moe_kernels.py`  (+91/-23)
- `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`  (+55/-36)
- `aiter/ops/flydsl/kernels/silu_and_mul_fq.py`  (+61/-5)
- `op_tests/test_moe_2stage.py`  (+39/-0)
- `csrc/include/rocm_ops.hpp`  (+20/-1)
- `aiter/ops/activation.py`  (+16/-0)
- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages.cu`  (+12/-0)
- `aiter/configs/model_configs/gptoss_fp4_tuned_fmoe.csv`  (+9/-0)
- `aiter/configs/model_configs/gptoss_fp4_untuned_fmoe.csv`  (+9/-0)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+9/-0)
- `csrc/include/activation.h`  (+9/-0)
- `aiter/aot/flydsl/moe.py`  (+6/-0)

## Key added lines (kernel files)

**`aiter/aot/flydsl/moe.py`**
```
enable_bias=enable_bias,
topk_ids = torch.zeros(tokens * topk, device=dev, dtype=torch.int32)
bias = torch.zeros(E * inter_dim * 2, device=dev, dtype=torch.float32)
topk_ids,
```

**`aiter/fused_moe.py`**
```
_USE_GENERIC_SWIGLU_MXFP4_LAYOUT = (
os.environ.get("GPTOSS_USE_GENERIC_SWIGLU_MXFP4_LAYOUT", "0") == "1"
_SWIGLU_MXFP4_BF16_BOUND = int(os.environ.get("GPTOSS_SWIGLU_MXFP4_BF16_BOUND", "256"))
isShuffled = getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False)
```

**`aiter/ops/activation.py`**
```
@compile_ops("module_activation", develop=True)
def swiglu_and_mul(out: Tensor, input: Tensor) -> None: ...
@compile_ops("module_activation", develop=True)
def silu_and_mul_bias(
```

**`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`**
```
def _load_bias_scalar(bias_rsrc, offset):
return buffer_ops.buffer_load(bias_rsrc, offset, vec_width=1, dtype=T.f32)
if const_expr(gate_up_interleave):
bias_pf = []
```

**`aiter/ops/flydsl/kernels/silu_and_mul_fq.py`**
```
"""Fused gate-activation-and-mul + quantization + sorted-scale write kernel (FlyDSL).
topk_ids (token_num * topk) i32, optional
bias     (expert, inter_dim * 2) f32, optional
act        : "silu" | "swiglu"
```
