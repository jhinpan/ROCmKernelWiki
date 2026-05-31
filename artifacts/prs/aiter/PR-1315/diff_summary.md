# Diff summary

- **files changed:** 184 (diff was byte-capped; summary is partial)
- **lines:** +1285 / -87
- **kernel-ish files:** 2

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/fused_mxfp4_quant.py`  (+159/-54)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=1280-K=8192.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=14336-K=8192.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=2560-K=8192.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=28672-K=8192.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=5120-K=8192.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=7168-K=8192.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=8192-K=1024.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=8192-K=14336.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=8192-K=2048.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=8192-K=3584.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=8192-K=4096.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=8192-K=7168.json`  (+86/-0)
- `aiter/ops/triton/activation.py`  (+9/-12)
- `aiter/ops/triton/configs/gemm/aot/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales_M=1-N=10240-K=8192/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales.json`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/fused_mxfp4_quant.py`**
```
rms_norm = row * norm_factor[:, None] * weight
@triton.heuristics(
"EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
and args["N1"] % (args["BLOCK_SIZE_N"]) == 0,
```

**`aiter/ops/triton/activation.py`**
```
scale_shuffle_padding: bool = False,
use_scale_shuffle_padding = shuffle or scale_shuffle_padding
if use_scale_shuffle_padding:
blockscale_e8m0 = torch.empty(
```
