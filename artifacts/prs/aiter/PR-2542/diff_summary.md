# Diff summary

- **files changed:** 9
- **lines:** +1625 / -792
- **kernel-ish files:** 8

## Files (by churn)

- `op_tests/op_benchmarks/triton/bench_mha.py`  (+578/-367)
- `aiter/ops/triton/attention/mha.py`  (+270/-172)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py`  (+267/-112)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_decode.py`  (+168/-72)
- `op_tests/triton_tests/attention/test_mha_dao_ai.py`  (+182/-0)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`  (+87/-30)
- `.github/workflows/flash_attention_integration.yaml`  (+54/-34)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py`  (+9/-5)
- `op_tests/triton_tests/attention/test_mha.py`  (+10/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py`**
```
AutotuneMode,
def get_bwd_configs(mode: AutotuneMode):
if mode == "off":
if arch.name == "gfx942":
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_decode.py`**
```
AutotuneMode,
def get_fwd_decode_configs(mode: AutotuneMode):
if mode == "off":
if arch.is_rdna:
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`**
```
AutotuneMode,
def get_fwd_prefill_configs(mode: AutotuneMode):
if mode == "off":
arch = get_arch()
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py`**
```
AutotuneMode = Literal["off", "on", "sweep"]
"AutotuneMode",
AUTOTUNE: AutotuneMode = (
if os.environ.get("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "1").lower()
```

**`aiter/ops/triton/attention/mha.py`**
```
from typing import Literal, Optional, Tuple, Union
_MHA_IMPL: Literal["default", "dao_ai"] = "default"
def mha_set_impl(impl: Literal["default", "dao_ai"]):
"""Set MHA forward implementation: 'default' (_attn_fwd) or 'dao_ai' (flash_attn_triton_amd)."""
```
