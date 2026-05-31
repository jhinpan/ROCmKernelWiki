# Diff summary

- **files changed:** 17
- **lines:** +3319 / -1367
- **kernel-ish files:** 11

## Files (by churn)

- `tile_engine/ops/gemm/gemm_instance_builder.py`  (+813/-737)
- `tile_engine/ops/gemm/gemm_benchmark.py`  (+721/-0)
- `tile_engine/ops/gemm/README.md`  (+412/-83)
- `tile_engine/ops/gemm/CMakeLists.txt`  (+261/-135)
- `tile_engine/ops/gemm/validation_utils.py`  (+342/-0)
- `tile_engine/ops/gemm/gemm_host_api.hpp`  (+0/-223)
- `tile_engine/ops/gemm/configs/default_config.json`  (+100/-100)
- `tile_engine/ops/gemm/gemm_common.hpp`  (+197/-0)
- `tile_engine/ops/gemm/benchmark_gemm_single.cpp`  (+160/-0)
- `tile_engine/ops/gemm/test_validation.py`  (+143/-0)
- `tile_engine/ops/gemm/test_benchmark.sh`  (+102/-0)
- `tile_engine/ops/gemm/benchmark_gemm.cpp`  (+0/-68)
- `tile_engine/ops/gemm/gemm_profiler.hpp`  (+32/-5)
- `tile_engine/ops/gemm/benchmark_gemm.hpp`  (+13/-6)
- `script/cmake-ck-dev.sh`  (+10/-3)

## Key added lines (kernel files)

**`tile_engine/ops/gemm/benchmark_gemm.hpp`**
```
<< "   \"layout_c\":\"" << problem.layout_c_ << "\",\n"
<< "   \"structured_sparsity\":" << (problem.structured_sparsity_ ? "true" : "false")
<< " \"name\": \"" << obj.name_ << "\",\n"
<< " \"problem\": " << obj.problem_ << ",\n"
```

**`tile_engine/ops/gemm/benchmark_gemm_single.cpp`**
```
inline auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("m", "3840", "The value for m dimension. Default is 3840.")
.insert("n", "4096", "The value for n dimension. Default is 4096.")
```

**`tile_engine/ops/gemm/codegen_utils.py`**
```
"fp8_bf8_fp16":  [
[16, 16, 128],
[32, 32, 64],
"bf8_fp8_fp16":  [
```

**`tile_engine/ops/gemm/gemm_benchmark.py`**
```
import sys
import json
import subprocess
import argparse
```

**`tile_engine/ops/gemm/gemm_common.hpp`**
```
template <typename T>
struct DataTypeTraits;
template <>
struct DataTypeTraits<float>
```
