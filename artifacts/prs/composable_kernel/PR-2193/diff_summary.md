# Diff summary

- **files changed:** 16
- **lines:** +1918 / -921
- **kernel-ish files:** 9

## Files (by churn)

- `tile_engine/ops/gemm/gemm_instance_builder.py`  (+421/-480)
- `tile_engine/ops/gemm/gemm_profiler.hpp`  (+260/-0)
- `tile_engine/ops/gemm/codegen_utils.py`  (+239/-0)
- `tile_engine/ops/gemm/benchmark_gemm.hpp`  (+233/-0)
- `tile_engine/ops/gemm/gemm_host_api.hpp`  (+76/-135)
- `tile_engine/ops/gemm/json_config.py`  (+202/-0)
- `tile_engine/ops/gemm/gemm_host_api.cpp`  (+0/-192)
- `tile_engine/ops/gemm/configs/default_config.json`  (+130/-0)
- `tile_engine/ops/gemm/configs/user_provided_config.json`  (+116/-0)
- `tile_engine/ops/gemm/benchmark_gemm.cpp`  (+68/-0)
- `tile_engine/ops/gemm/README.md`  (+35/-32)
- `tile_engine/ops/gemm/configs/instance_combination.json`  (+0/-62)
- `include/ck_tile/host/device_prop.hpp`  (+56/-0)
- `tile_engine/ops/gemm/CMakeLists.txt`  (+34/-19)
- `Jenkinsfile`  (+47/-1)

## Key added lines (kernel files)

**`include/ck_tile/host/device_prop.hpp`**
```
namespace ck_tile {
constexpr unsigned int fnv1a_hash(std::string_view str, unsigned int h = 2166136261u)
return str.empty() ? h
: fnv1a_hash(str.substr(1),
```

**`tile_engine/ops/gemm/benchmark_gemm.cpp`**
```
void benchmark_gemm(const ck_tile::ArgParser& arg_parser)
GemmProblem gemm_problem{arg_parser.get_int("split_k"),
arg_parser.get_int("m"),
arg_parser.get_int("n"),
```

**`tile_engine/ops/gemm/benchmark_gemm.hpp`**
```
enum class Metric
LATENCY   = 0,
TFLOPS    = 1,
BANDWIDTH = 2
```

**`tile_engine/ops/gemm/codegen_utils.py`**
```
Mappings and utility functions for kernel code generation.
import subprocess
import re
from functools import lru_cache
```

**`tile_engine/ops/gemm/gemm_host_api.hpp`**
```
arg_parser.insert("m", "3840", "The value for m dimension. Default is 3840.")
.insert("n", "4096", "The value for n dimension. Default is 4096.")
.insert("k", "2048", "The value for k dimension. Default is 2048.")
.insert("stride_a", "0", "The stride value for tensor A. Default is 0.")
```
