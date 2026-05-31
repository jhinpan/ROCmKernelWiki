# Diff summary

- **files changed:** 10
- **lines:** +229 / -1165
- **kernel-ish files:** 8

## Files (by churn)

- `python/perf-kernels/tune_gemm/tune_gemm.py`  (+114/-333)
- `python/perf-kernels/tune_gemm/matmul.py`  (+0/-375)
- `python/perf-kernels/tune_gemm/rocprof_gemm.py`  (+0/-318)
- `python/perf-kernels/tune_gemm/one_config.py`  (+43/-31)
- `python/perf-kernels/tune_gemm/README.md`  (+33/-17)
- `python/perf-kernels/tune_gemm/utils/file_generator.py`  (+21/-22)
- `python/perf-kernels/tune_gemm/icache_flush.py`  (+8/-20)
- `python/perf-kernels/tune_gemm/tune_gemm.sh`  (+0/-27)
- `python/perf-kernels/tune_gemm/utils/utils.py`  (+6/-11)
- `python/perf-kernels/tune_gemm/matmul_kernel.py`  (+4/-11)

## Key added lines (kernel files)

**`python/perf-kernels/tune_gemm/icache_flush.py`**
```
elif (isinstance(err, hiprtc.hiprtcResult) and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS):
cflags = [b"--offload-arch=" + arch]
hip_check(
hip.hipModuleLaunchKernel(kernel, *grid, *block, sharedMemBytes=0, stream=None, kernelParams=None, extra=()))
```

**`python/perf-kernels/tune_gemm/matmul_kernel.py`**
```
def matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
stride_cn, stride_bias, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr,
EVEN_K: tl.constexpr):
```

**`python/perf-kernels/tune_gemm/one_config.py`**
```
parser.add_argument("--init_type", type=str, default='randn',
help="Initialization type for input matrices (default uniform rand [0, 1.0)])")
parser.add_argument(
"--config_str", type=str, default="", help=
```

**`python/perf-kernels/tune_gemm/tune_gemm.py`**
```
from utils.file_generator import (
gen_configStr,
generate_compile_driver,
generate_matmul_kernels,
```

**`python/perf-kernels/tune_gemm/utils/file_generator.py`**
```
from .utils import (
get_filename_compile_driver,
get_filename_myKernels,
get_filename_profile_driver,
```
