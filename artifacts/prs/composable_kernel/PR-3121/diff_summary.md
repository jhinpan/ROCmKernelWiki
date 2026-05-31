# Diff summary

- **files changed:** 22
- **lines:** +2415 / -1974
- **kernel-ish files:** 13

## Files (by churn)

- `tile_engine/ops/gemm_multi_d/gemm_multi_d_instance_builder.py`  (+791/-643)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_benchmark.py`  (+683/-0)
- `tile_engine/ops/gemm_multi_d/CMakeLists.txt`  (+277/-141)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_config.py`  (+0/-250)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_codegen_utils.py`  (+0/-196)
- `tile_engine/ops/gemm_multi_d/configs/default_config.json`  (+101/-81)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_benchmark_single.cpp`  (+170/-0)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_host_api.hpp`  (+0/-164)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_benchmark.hpp`  (+87/-73)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_profiler.hpp`  (+84/-55)
- `tile_engine/ops/gemm_multi_d/README.md`  (+0/-110)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_common.hpp`  (+100/-0)
- `tile_engine/ops/gemm_multi_d/configs/custom_ci_config.json`  (+0/-80)
- `tile_engine/ops/gemm_multi_d/benchmark_gemm_multi_d.cpp`  (+0/-73)
- `tile_engine/ops/commons/validation_utils.py`  (+23/-36)

## Key added lines (kernel files)

**`tile_engine/ops/commons/validation_utils.py`**
```
"gfx1201": {  # Check how to handle for GEMM and Multi D
def get_abcd_layouts(layout_code: str) -> Tuple[str, str, str, List[str]]:
Return (ALayout, BLayout, CLayout) from a 3-letter code like 'rcrr', 'ccrr', 'crrr', 'rrrr'.
code = str(layout_code).strip().lower()
```

**`tile_engine/ops/gemm/gemm_instance_builder.py`**
```
import importlib.util
def _import_validation_utils():
"""Import validation utilities from commons directory."""
current_dir = os.path.dirname(os.path.abspath(__file__))
```

**`tile_engine/ops/gemm_multi_d/gemm_multi_d_benchmark.hpp`**
```
enum class Metric
LATENCY   = 0,
TFLOPS    = 1,
BANDWIDTH = 2
```

**`tile_engine/ops/gemm_multi_d/gemm_multi_d_benchmark.py`**
```
import sys
import json
import subprocess
import argparse
```

**`tile_engine/ops/gemm_multi_d/gemm_multi_d_benchmark_single.cpp`**
```
inline auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("m", "3840", "The value for m dimension. Default is 3840.")
.insert("n", "4096", "The value for n dimension. Default is 4096.")
```
