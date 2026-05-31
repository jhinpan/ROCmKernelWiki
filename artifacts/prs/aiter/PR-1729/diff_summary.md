# Diff summary

- **files changed:** 8 (diff was byte-capped; summary is partial)
- **lines:** +395 / -5
- **kernel-ish files:** 1

## Files (by churn)

- `aiter/configs/tuned_fmoe.csv`  (+284/-0)
- `aiter/aot/sampling.py`  (+89/-0)
- `.github/workflows/pre-checks.yaml`  (+11/-2)
- `MANIFEST.in`  (+5/-1)
- `.github/workflows/triton-test.yaml`  (+1/-1)
- `3rdparty/composable_kernel`  (+1/-1)
- `aiter/configs/a4w4_blockscale_tuned_gemm.csv`  (+2/-0)
- `aiter/configs/a4w4_blockscale_untuned_gemm.csv`  (+2/-0)

## Key added lines (kernel files)

**`aiter/aot/sampling.py`**
```
from collections import namedtuple
import os
import concurrent.futures
from csrc.cpp_itfs.sampling.top_k_renorm_probs import (
```
