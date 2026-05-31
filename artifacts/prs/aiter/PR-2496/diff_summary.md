# Diff summary

- **files changed:** 12 (diff was byte-capped; summary is partial)
- **lines:** +1165 / -530
- **kernel-ish files:** 1

## Files (by churn)

- `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`  (+479/-479)
- `aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3.5_397b.csv`  (+561/-0)
- `aiter/__init__.py`  (+52/-40)
- `.github/workflows/flash_attention_integration.yaml`  (+63/-3)
- `.github/workflows/sglang_downstream.yaml`  (+4/-0)
- `aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv`  (+0/-4)
- `.github/scripts/build_aiter_triton.sh`  (+2/-1)
- `.github/workflows/atom-test.yaml`  (+1/-1)
- `.github/workflows/vllm_benchmark.yaml`  (+1/-1)
- `3rdparty/composable_kernel`  (+1/-1)
- `.github/workflows/aiter-test.yaml`  (+1/-0)
- `aiter/configs/model_configs/dsv3_bf16_untuned_gemm.csv`  (+0/-0)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
import sys
if sys.platform == "win32":
logger.info("Windows: CK and HIP ops are not available. Triton ops only.")
from .jit import core as core  # noqa: E402
```
