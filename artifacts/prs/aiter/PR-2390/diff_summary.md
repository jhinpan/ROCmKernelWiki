# Diff summary

- **files changed:** 10 (diff was byte-capped; summary is partial)
- **lines:** +2190 / -1560
- **kernel-ish files:** 9

## Files (by churn)

- `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`  (+1233/-1026)
- `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`  (+306/-283)
- `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`  (+333/-223)
- `aiter/ops/flydsl/kernels/layout_utils.py`  (+128/-0)
- `aiter/fused_moe.py`  (+83/-1)
- `aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv`  (+22/-17)
- `aiter/ops/flydsl/kernels/kernels_common.py`  (+34/-0)
- `aiter/ops/flydsl/kernels/mfma_epilogues.py`  (+28/-4)
- `aiter/ops/flydsl/__init__.py`  (+22/-2)
- `aiter/ops/flydsl/kernels/__init__.py`  (+1/-4)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
def _flydsl_stage1_wrapper(
hidden_states,
sorted_token_ids,
sorted_expert_ids,
```

**`aiter/ops/flydsl/__init__.py`**
```
Kernel compilation and public APIs are only available when a compatible
``flydsl`` package is installed. Use ``is_flydsl_available()`` to check
whether the optional dependency exists before relying on FlyDSL kernels.
from importlib.metadata import PackageNotFoundError, version
```

**`aiter/ops/flydsl/kernels/__init__.py`**
```
"""FlyDSL MOE kernel builders (stage1, stage2, reduction)."""
```

**`aiter/ops/flydsl/kernels/kernels_common.py`**
```
"""Common helpers shared by kernel modules.
Keep helper naming consistent with other kernel helpers (e.g. `mfma_preshuffle_pipeline.py`),
but this module is intentionally small and MLIR-dialect facing.
from flydsl._mlir import ir
```

**`aiter/ops/flydsl/kernels/layout_utils.py`**
```
"""Layout helpers for GEMM kernels.
Parses fly layout type strings (e.g. '(4,64):(64,1)') and computes
idx2crd / crd2idx with plain arith ops for static layouts.
Falls back to fly dialect ops for dynamic layouts.
```
