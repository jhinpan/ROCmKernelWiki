# Diff summary

- **files changed:** 73
- **lines:** +7581 / -55
- **kernel-ish files:** 6

## Files (by churn)

- `op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py`  (+5872/-0)
- `aiter/ops/triton/gemm_afp4wfp4.py`  (+436/-24)
- `aiter/utility/triton/README.md`  (+186/-0)
- `aiter/utility/triton/triton_metadata_redirect.py`  (+166/-0)
- `op_tests/triton_tests/test_gemm_afp4wfp4.py`  (+91/-19)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED.json`  (+87/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=10240-K=8192.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=106496-K=16384.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=16384-K=16384.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=16384-K=53248.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=18432-K=16384.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=57344-K=8192.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=8192-K=28672.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=8192-K=8192.json`  (+86/-0)
- `op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`  (+13/-12)

## Key added lines (kernel files)

**`aiter/ops/triton/gemm_afp4wfp4.py`**
```
import os
from aiter.utility.triton.triton_metadata_redirect import AOTMetadataContext
GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
```

**`aiter/utility/triton/triton_metadata_redirect.py`**
```
Triton Metadata Redirect Module
This module provides decorators and utilities for customizing Triton kernel
metadata file paths during compilation. It allows redirecting .json and
.hsaco files to custom directories.
```

**`op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`**
```
gemm_afp4wfp4_preshuffled_weight_scales,
def bench_gemm_fn(M: int, N: int, K: int, metric: str, layout: str, shuffle: bool):
x, _, w, _, _, x_scale, w_scale, _, y = generate_gemm_afp4wfp4_inputs(
shuffle_scales_fg=shuffle,
```

**`op_tests/triton_tests/test_gemm_afp4wfp4.py`**
```
gemm_afp4wfp4_preshuffled_weight_scales,
from aiter.ops.shuffle import shuffle_weight
def generate_gemm_afp4wfp4_inputs(
layout="TN",
```

**`op_tests/triton_tests/triton_metadata_redirect/kernel.py`**
```
import triton
import triton.language as tl
@triton.jit
def empty_kernel(x_ptr, SIZE: tl.constexpr):
```
