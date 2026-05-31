# Diff summary

- **files changed:** 12 (diff was byte-capped; summary is partial)
- **lines:** +6483 / -0
- **kernel-ish files:** 9

## Files (by churn)

- `aiter/ops/triton/gluon/pa_decode_gluon.py`  (+2815/-0)
- `csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py`  (+1121/-0)
- `csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot_prebuild.py`  (+1038/-0)
- `csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot.py`  (+775/-0)
- `csrc/cpp_itfs/gluon_aot_tools/compile_gluon.py`  (+383/-0)
- `csrc/cpp_itfs/pa_gluon_aot/pa_decode_attention_reduce_kernel.cpp.jinja`  (+111/-0)
- `csrc/cpp_itfs/pa_gluon_aot/pa_decode_attention_kernel.cpp.jinja`  (+96/-0)
- `csrc/cpp_itfs/gluon_aot_tools/extra/hip/compile.cpp`  (+67/-0)
- `csrc/cpp_itfs/pa_gluon_aot/pa_decode_reduce_kernel.cpp.jinja`  (+52/-0)
- `csrc/cpp_itfs/gluon_aot_tools/extra/hip/compile.h`  (+13/-0)
- `csrc/cpp_itfs/pa_gluon_aot/pa_reduce_kernel_test.py`  (+10/-0)
- `aiter/ops/triton/gluon/__init__.py`  (+2/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/gluon/pa_decode_gluon.py`**
```
import os
import sys
import math
from typing import Optional, Dict, Tuple
```

**`csrc/cpp_itfs/gluon_aot_tools/compile_gluon.py`**
```
import binascii
import hashlib
import importlib.util
import sys
```

**`csrc/cpp_itfs/gluon_aot_tools/extra/hip/compile.cpp`**
```
gpuAssert((ans), __FILE__, __LINE__);\
static inline void gpuAssert(hipError_t code, const char *file, int line) {{
if (code != hipSuccess) {{
const char *prefix = "Triton Error [HIP]: ";
```

**`csrc/cpp_itfs/gluon_aot_tools/extra/hip/compile.h`**
```
void unload_{kernel_name}(void);
void load_{kernel_name}(void);
hipError_t{_placeholder} {kernel_name}(hipStream_t stream, {signature});
```

**`csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py`**
```
import os
import sys
import hashlib
import aiter
```
