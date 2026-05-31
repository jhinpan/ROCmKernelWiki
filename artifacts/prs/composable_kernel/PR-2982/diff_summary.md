# Diff summary

- **files changed:** 18
- **lines:** +510 / -993
- **kernel-ish files:** 9

## Files (by churn)

- `tile_engine/ops/gemm/gemm_instance_builder.py`  (+232/-344)
- `tile_engine/ops/gemm/commons/validation_utils.py`  (+250/-0)
- `tile_engine/ops/gemm/json_config.py`  (+0/-231)
- `tile_engine/ops/gemm/configs/benchmark.json`  (+0/-105)
- `tile_engine/ops/gemm/configs/gfx120x_config.json`  (+0/-102)
- `tile_engine/ops/gemm/configs/custom_ci_config.json`  (+0/-88)
- `tile_engine/ops/gemm/gemm_common.hpp`  (+1/-46)
- `tile_engine/ops/gemm/gemm_benchmark.py`  (+0/-42)
- `tile_engine/ops/gemm/configs/user_provided_config.json`  (+12/-18)
- `tile_engine/ops/gemm/CMakeLists.txt`  (+7/-7)
- `tile_engine/ops/gemm/gemm_benchmark_single.cpp`  (+4/-4)
- `tile_engine/ops/gemm/configs/default_config.json`  (+2/-3)
- `tile_engine/ops/gemm/README.md`  (+1/-1)
- `tile_engine/ops/gemm/gemm_profiler.hpp`  (+1/-1)
- `Jenkinsfile`  (+0/-1)

## Key added lines (kernel files)

**`tile_engine/ops/gemm/commons/validation_utils.py`**
```
WARP_SUPPORTED_COMBINATIONS = {
"gfx90a": [
[1, 4, 1],
[2, 2, 1],
```

**`tile_engine/ops/gemm/gemm_benchmark_single.cpp`**
```
"for validation on GPU. Default is 2, GPU validation.")
void benchmark_single(const ck_tile::ArgParser& arg_parser)
benchmark_single(parser);
```

**`tile_engine/ops/gemm/gemm_instance_builder.py`**
```
from commons.validation_utils import (
is_tile_config_valid,
is_trait_combination_valid,
get_dtype_string,
```
