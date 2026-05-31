# Diff summary

- **files changed:** 18
- **lines:** +2553 / -297
- **kernel-ish files:** 10

## Files (by churn)

- `tile_engine/ops/gemm_multi_d/gemm_multi_d_instance_builder.py`  (+755/-0)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_profiler.hpp`  (+278/-0)
- `include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`  (+51/-203)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_config.py`  (+250/-0)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_codegen_utils.py`  (+229/-0)
- `tile_engine/ops/gemm_multi_d/benchmark_gemm_multi_d.hpp`  (+218/-0)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_host_api.hpp`  (+164/-0)
- `tile_engine/ops/gemm_multi_d/CMakeLists.txt`  (+152/-0)
- `tile_engine/ops/gemm_multi_d/README.md`  (+110/-0)
- `example/ck_tile/19_gemm_multi_d/gemm_multi_d_fp16.cpp`  (+1/-89)
- `tile_engine/ops/gemm_multi_d/configs/default_config.json`  (+84/-0)
- `tile_engine/ops/gemm_multi_d/configs/user_provided_config.json`  (+81/-0)
- `tile_engine/ops/gemm_multi_d/configs/custom_ci_config.json`  (+80/-0)
- `tile_engine/ops/gemm_multi_d/benchmark_gemm_multi_d.cpp`  (+73/-0)
- `Jenkinsfile`  (+22/-2)

## Key added lines (kernel files)

**`example/ck_tile/19_gemm_multi_d/gemm_multi_d_fp16.cpp`**
```
BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
```

**`include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`**
```
template <class T>
using raw_t = std::remove_cv_t<std::remove_reference_t<T>>;
template <class Y, class X>
CK_TILE_HOST_DEVICE void operator()(Y&& y, const X& x) const
```

**`tile_engine/ops/gemm_multi_d/benchmark_gemm_multi_d.cpp`**
```
void benchmark_gemm_multi_d(const ck_tile::ArgParser& arg_parser)
GemmMultiDProblem gemm_multi_d_problem{arg_parser.get_int("split_k"),
arg_parser.get_int("m"),
arg_parser.get_int("n"),
```

**`tile_engine/ops/gemm_multi_d/benchmark_gemm_multi_d.hpp`**
```
struct GemmMultiDProblem
int split_k_;
int m_, n_, k_;
int stride_a_, stride_b_, stride_d0_, stride_d1_, stride_e_;
```

**`tile_engine/ops/gemm_multi_d/gemm_multi_d_codegen_utils.py`**
```
Mappings and utility functions for kernel code generation.
import subprocess
import re
from functools import lru_cache
```
