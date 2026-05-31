# Diff summary

- **files changed:** 41
- **lines:** +2243 / -3455
- **kernel-ish files:** 26

## Files (by churn)

- `tile_engine/ops/gemm/gemm_instance_builder.py`  (+572/-433)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_instance_builder.py`  (+0/-894)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_instance_builder.py`  (+0/-891)
- `tile_engine/ops/gemm/README.md`  (+0/-442)
- `tile_engine/ops/gemm/gemm_multi_d/gemm_multi_d_instance_builder.py`  (+330/-0)
- `tile_engine/ops/gemm/CMakeLists.txt`  (+3/-307)
- `tile_engine/ops/gemm/gemm_universal/CMakeLists.txt`  (+309/-0)
- `tile_engine/ops/gemm/gemm_preshuffle/gemm_preshuffle_instance_builder.py`  (+300/-0)
- `tile_engine/ops/gemm/gemm_universal/gemm_universal_instance_builder.py`  (+295/-0)
- `tile_engine/ops/gemm/gemm_preshuffle/gemm_preshuffle_common.hpp`  (+181/-0)
- `tile_engine/ops/commons/test_validation.py`  (+0/-144)
- `tile_engine/ops/commons/test_benchmark.sh`  (+0/-105)
- `tile_engine/ops/gemm/gemm_multi_d/gemm_multi_d_common.hpp`  (+100/-0)
- `tile_engine/ops/gemm/gemm_universal/gemm_common.hpp`  (+100/-0)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_common.hpp`  (+0/-83)

## Key added lines (kernel files)

**`tile_engine/ops/gemm/gemm_instance_builder.py`**
```
import itertools
import logging
os.path.join(parent_dir, "gemm", "gemm_validation_utils.py"),
get_abcd_layouts = _validation_utils.get_abcd_layouts
```

**`tile_engine/ops/gemm/gemm_multi_d/gemm_multi_d_benchmark_single.cpp`**
```
std::string dtype_a   = DataTypeTraits<ADataType>::name;
std::string dtype_b   = DataTypeTraits<BDataType>::name;
std::string dtype_acc = DataTypeTraits<AccDataType>::name;
std::string dtype_c   = DataTypeTraits<CDataType>::name;
```

**`tile_engine/ops/gemm/gemm_multi_d/gemm_multi_d_common.hpp`**
```
template <typename T>
struct DataTypeTraits;
template <>
struct DataTypeTraits<float>
```

**`tile_engine/ops/gemm/gemm_multi_d/gemm_multi_d_instance_builder.py`**
```
import os
import argparse
import importlib.util
import multiprocessing
```

**`tile_engine/ops/gemm/gemm_preshuffle/gemm_preshuffle_benchmark_single.cpp`**
```
std::string dtype_a   = DataTypeTraits<ADataType>::name;
std::string dtype_b   = DataTypeTraits<BDataType>::name;
std::string dtype_acc = DataTypeTraits<AccDataType>::name;
std::string dtype_c   = DataTypeTraits<CDataType>::name;
```
