# Diff summary

- **files changed:** 29
- **lines:** +1944 / -741
- **kernel-ish files:** 20

## Files (by churn)

- `tests/kernels/utils/fp4_utils.py`  (+687/-0)
- `kernels/preshuffle_gemm_flyc.py`  (+354/-172)
- `README.md`  (+99/-205)
- `tests/kernels/test_preshuffle_gemm.py`  (+199/-23)
- `scripts/run_tests.sh`  (+29/-191)
- `setup.py`  (+51/-82)
- `scripts/build.sh`  (+126/-0)
- `python/flydsl/expr/utils/arith.py`  (+101/-1)
- `scripts/run_benchmark.sh`  (+69/-6)
- `kernels/layout_utils.py`  (+67/-0)
- `python/mlir_flydsl/CMakeLists.txt`  (+34/-1)
- `tests/test_common.py`  (+26/-2)
- `scripts/test_flyc.sh`  (+0/-25)
- `kernels/mfma_preshuffle_pipeline.py`  (+13/-9)
- `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`  (+21/-0)

## Key added lines (kernel files)

**`flir/python_bindings/runtime/FlirRocmRuntimeWrappers.cpp`**
```
static thread_local hipStream_t tls_capture_stream = nullptr;
extern "C" void mgpuSetCaptureStream(void* stream) {
tls_capture_stream = (hipStream_t)stream;
if (tls_capture_stream)
```

**`kernels/layout_utils.py`**
```
"""Pure-arith layout helpers for static-stride layouts.
Parses fly layout type strings (e.g. '(4,64):(64,1)') and computes
idx2crd / crd2idx with plain arith ops, avoiding fly dialect round-trips.
import re
```

**`kernels/mfma_preshuffle_pipeline.py`**
```
from kernels.layout_utils import crd2idx, idx2crd, get as layout_get
i32_val = buffer_ops.buffer_load(rsrc, idx_i32, vec_width=vec_width, dtype=T.i32)
if vec_width == 1:
i32_vec = vector.from_elements(T.vec(1, T.i32), [i32_val])
```

**`kernels/preshuffle_gemm_flyc.py`**
```
from kernels.layout_utils import crd2idx, idx2crd, get as layout_get
waves_per_eu: int = None,
use_async_copy: bool = False,
Signature:  launch_fn(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, M, N, stream)
```

**`lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`**
```
class ExtractAlignedPointerAsIndexLowering
: public OpConversionPattern<ExtractAlignedPointerAsIndexOp> {
using OpConversionPattern::OpConversionPattern;
LogicalResult matchAndRewrite(ExtractAlignedPointerAsIndexOp op, OpAdaptor adaptor,
```
