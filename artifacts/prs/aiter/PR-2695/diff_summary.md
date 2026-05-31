# Diff summary

- **files changed:** 17
- **lines:** +124 / -267
- **kernel-ish files:** 12

## Files (by churn)

- `aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py`  (+23/-61)
- `aiter/ops/triton/gluon/pa_decode_gluon.py`  (+14/-46)
- `.github/workflows/triton-test.yaml`  (+4/-38)
- `.github/scripts/build_aiter_triton.sh`  (+8/-26)
- `aiter/ops/triton/gluon/triton_version.py`  (+0/-30)
- `.github/requirements/triton-test.txt`  (+25/-0)
- `aiter/ops/triton/gluon/gemm_a8w8.py`  (+6/-17)
- `.github/scripts/verify_triton_pin.py`  (+17/-0)
- `.github/scripts/install_triton.sh`  (+0/-14)
- `aiter/ops/triton/gluon/__init__.py`  (+14/-0)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py`  (+0/-13)
- `op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py`  (+0/-13)
- `aiter/ops/triton/gluon/gemm_afp4wfp4.py`  (+1/-8)
- `.github/workflows/flash_attention_integration.yaml`  (+5/-0)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v2.py`  (+5/-0)

## Key added lines (kernel files)

**`.github/scripts/verify_triton_pin.py`**
```
"""Assert installed triton matches the pin in the given requirements file."""
import re
import sys
from importlib.metadata import version
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py`**
```
pT_dropout = pT * dropout_mask.to(pT.dtype) * dropout_scale
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v2.py`**
```
num_splits: int = 0,
if num_splits not in (0, 1):
raise NotImplementedError(
"num_splits > 1 not supported in AMD Triton FA2 varlen_fwd."
```

**`aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py`**
```
_gemm_afp4wfp4_preshuffle_kernel[grid](
y if config["NUM_KSPLIT"] == 1 else y_pp,
x_scales,
w_scales,
```

**`aiter/ops/triton/gluon/__init__.py`**
```
from packaging.version import Version
import triton
_triton_version = Version(triton.__version__.split("+")[0])
_min_version = Version("3.6.0")
```
