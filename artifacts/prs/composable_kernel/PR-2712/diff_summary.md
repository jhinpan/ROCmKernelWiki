# Diff summary

- **files changed:** 12
- **lines:** +3256 / -1
- **kernel-ish files:** 7

## Files (by churn)

- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_instance_builder.py`  (+836/-0)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_benchmark.py`  (+684/-0)
- `tile_engine/ops/gemm_preshuffle/commons/validation_utils.py`  (+375/-0)
- `tile_engine/ops/gemm_preshuffle/CMakeLists.txt`  (+296/-0)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_profiler.hpp`  (+275/-0)
- `tile_engine/ops/gemm_preshuffle/benchmark_gemm_preshuffle.hpp`  (+225/-0)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_common.hpp`  (+213/-0)
- `tile_engine/ops/gemm_preshuffle/benchmark_gemm_preshuffle_single.cpp`  (+164/-0)
- `tile_engine/ops/gemm_preshuffle/configs/default_config.json`  (+90/-0)
- `tile_engine/ops/gemm_preshuffle/configs/user_provided_config.json`  (+86/-0)
- `Jenkinsfile`  (+10/-0)
- `tile_engine/ops/CMakeLists.txt`  (+2/-1)

## Key added lines (kernel files)

**`tile_engine/ops/gemm_preshuffle/benchmark_gemm_preshuffle.hpp`**
```
enum class Metric
LATENCY   = 0,
TFLOPS    = 1,
BANDWIDTH = 2
```

**`tile_engine/ops/gemm_preshuffle/benchmark_gemm_preshuffle_single.cpp`**
```
inline auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("m", "3840", "The value for m dimension. Default is 3840.")
.insert("n", "4096", "The value for n dimension. Default is 4096.")
```

**`tile_engine/ops/gemm_preshuffle/commons/validation_utils.py`**
```
Validation utilities for GEMM kernel generation.
Extracted from tile_engine_develop for consistency.
import subprocess
import re
```

**`tile_engine/ops/gemm_preshuffle/gemm_preshuffle_benchmark.py`**
```
import sys
import json
import subprocess
import argparse
```

**`tile_engine/ops/gemm_preshuffle/gemm_preshuffle_common.hpp`**
```
template <typename T>
struct DataTypeTraits;
template <>
struct DataTypeTraits<float>
```
