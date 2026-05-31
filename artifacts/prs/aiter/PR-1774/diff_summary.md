# Diff summary

- **files changed:** 12 (diff was byte-capped; summary is partial)
- **lines:** +1570 / -2119
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/triton/gluon/pa_decode_gluon.py`  (+777/-830)
- `csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot_prebuild.py`  (+298/-327)
- `csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py`  (+221/-229)
- `csrc/cpp_itfs/pa_gluon_aot/transpose_query_output_gluon_aot.py`  (+0/-376)
- `csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot.py`  (+94/-123)
- `csrc/cpp_itfs/pa_gluon_aot/pa_reduce_kernel_test.py`  (+93/-80)
- `csrc/cpp_itfs/pa_gluon_aot/pa_decode_attention_reduce_kernel.cpp.jinja`  (+41/-26)
- `csrc/cpp_itfs/pa_gluon_aot/transpose_output_gluon_kernel.cpp.jinja`  (+0/-49)
- `csrc/cpp_itfs/pa_gluon_aot/transpose_query_gluon_kernel.cpp.jinja`  (+0/-49)
- `csrc/cpp_itfs/pa_gluon_aot/pa_decode_attention_kernel.cpp.jinja`  (+26/-17)
- `csrc/cpp_itfs/pa_gluon_aot/pa_decode_reduce_kernel.cpp.jinja`  (+18/-11)
- `aiter/ops/attention.py`  (+2/-2)

## Key added lines (kernel files)

**`aiter/ops/attention.py`**
```
["output", "exp_sums", "max_logits", "temporary_output"],
```

**`aiter/ops/triton/gluon/pa_decode_gluon.py`**
```
def parse_triton_version(version_str):
"""Parse version string into comparable tuple format, handling possible development version suffixes"""
version_str = version_str.split("+")[0].split("-")[0]
parts = []
```

**`csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py`**
```
from aiter.test_common import perftest
import triton
import triton.language as tl
import aiter.ops.triton.utils._triton.arch_info as arch_info
```

**`csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot.py`**
```
from aiter.ops.triton.gluon.pa_decode_gluon import get_cdna_version
compute_type: torch.dtype,
query_seq_len: int,
one_query_group_size: int,
```

**`csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot_prebuild.py`**
```
import sys
from typing import Optional, Tuple, Union, Dict
import multiprocessing
import concurrent.futures
```
