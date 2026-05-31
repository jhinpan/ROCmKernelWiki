# Diff summary

- **files changed:** 22
- **lines:** +523 / -407
- **kernel-ish files:** 17

## Files (by churn)

- `test/ck_tile/gemm_streamk_tile_engine/generate_configs.py`  (+277/-0)
- `test/ck_tile/gemm_streamk_tile_engine/generate_configs.cmake`  (+103/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_reduction_cases.inc`  (+0/-88)
- `test/ck_tile/gemm_streamk_tile_engine/CMakeLists.txt`  (+33/-21)
- `test/ck_tile/gemm_streamk_tile_engine/test_gemm_streamk_simple.cpp`  (+33/-20)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_smoke_cases.inc`  (+0/-47)
- `test/ck_tile/gemm_streamk_tile_engine/cu_count.cpp`  (+44/-0)
- `test/ck_tile/gemm_streamk_tile_engine/configs/simple_test_config.json`  (+0/-35)
- `tile_engine/ops/gemm_streamk/gemm_streamk_profiler.hpp`  (+11/-12)
- `test/ck_tile/gemm_streamk_tile_engine/README.md`  (+14/-6)
- `test/ck_tile/gemm_streamk/smoke_tests/test_gemm_streamk_bf16_nonpersistent.cpp`  (+0/-17)
- `test/ck_tile/gemm_streamk/smoke_tests/test_gemm_streamk_bf16_persistent.cpp`  (+0/-17)
- `test/ck_tile/gemm_streamk/smoke_tests/test_gemm_streamk_bf8_nonpersistent.cpp`  (+0/-17)
- `test/ck_tile/gemm_streamk/smoke_tests/test_gemm_streamk_bf8_persistent.cpp`  (+0/-17)
- `test/ck_tile/gemm_streamk/smoke_tests/test_gemm_streamk_fp16_nonpersistent.cpp`  (+0/-17)

## Key added lines (kernel files)

**`test/ck_tile/gemm_streamk_tile_engine/cu_count.cpp`**
```
bool has_error(const hipError_t& error_status)
if(error_status != hipSuccess)
std::cerr << hipGetErrorString(error_status);
return true;
```

**`test/ck_tile/gemm_streamk_tile_engine/generate_configs.py`**
```
from enum import Enum
from typing import Dict, Tuple, List
import argparse
import json
```

**`test/ck_tile/gemm_streamk_tile_engine/test_gemm_streamk_simple.cpp`**
```
EXPECT_TRUE(strlen(KERNEL_NAME) > 0) << "Kernel name should not be empty";
std::cout << "Testing kernel: " << KERNEL_NAME << std::endl;
std::cout << "Problem size: " << m_ << "x" << n_ << "x" << k_ << std::endl;
ck_tile::HostTensor<CDataType> c_m_n_dev_ref(
```

**`tile_engine/ops/gemm_streamk/gemm_streamk_instance_builder.py`**
```
static std::tuple<float, ck_tile::index_t> launch(const ck_tile::StreamKHostArgs& args,
const ck_tile::stream_config& stream) {{
const ck_tile::index_t num_wgs_per_tile = kargs.tile_partitioner.estimate_num_wgs_per_tile();
const float time = ck_tile::launch_kernel_time_mask(
```

**`tile_engine/ops/gemm_streamk/gemm_streamk_profiler.hpp`**
```
std::function<std::tuple<float, ck_tile::index_t>(
const ck_tile::StreamKHostArgs&, const ck_tile::stream_config&)> kernel_func)
std::vector<std::function<std::tuple<std::string, float, ck_tile::index_t>(
ck_tile::StreamKHostArgs&, const ck_tile::stream_config&)>>
```
