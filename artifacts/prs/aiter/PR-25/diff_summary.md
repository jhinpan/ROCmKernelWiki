# Diff summary

- **files changed:** 14
- **lines:** +680 / -232
- **kernel-ish files:** 11

## Files (by churn)

- `csrc/ck_gemm_a8w8/gen_instances.py`  (+129/-167)
- `csrc/ck_gemm_a8w8/gemm_a8w8_common.py`  (+150/-0)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`  (+136/-0)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.cu`  (+98/-0)
- `csrc/ck_gemm_a8w8/include/gemm_a8w8_common.cuh`  (+24/-24)
- `csrc/ck_gemm_a8w8/gemm_a8w8.cu`  (+21/-12)
- `ater/configs/a8w8_tuned_gemm.csv`  (+27/-0)
- `ater/configs/a8w8_untuned_gemm.csv`  (+27/-0)
- `ater/ops/gemm_op_a8w8.py`  (+22/-5)
- `setup.py`  (+6/-17)
- `csrc/ck_gemm_a8w8/README.md`  (+17/-0)
- `op_tests/test_gemm_a8w8.py`  (+7/-7)
- `csrc/ck_gemm_a8w8/include/gemm_a8w8.h`  (+9/-0)
- `csrc/pybind/gemm_a8w8_tune_pybind.cu`  (+7/-0)

## Key added lines (kernel files)

**`ater/ops/gemm_op_a8w8.py`**
```
srcs=[f"{ATER_CSRC_DIR}/pybind/gemm_a8w8_pybind.cu",
f"{ATER_CSRC_DIR}/ck_gemm_a8w8/gemm_a8w8.cu",
f"{ATER_CSRC_DIR}/ck_gemm_a8w8/include",],
blob_gen_cmd =  f"{ATER_CSRC_DIR}/ck_gemm_a8w8/gen_instances.py --working_path {{}} --tune_file ater/configs/a8w8_tuned_
```

**`csrc/ck_gemm_a8w8/gemm_a8w8.cu`**
```
static const auto lookup = []
if constexpr (std::is_same_v<EDataType, F16>) {
return RowwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType,F16)};
} else if constexpr (std::is_same_v<EDataType, B16>) {
```

**`csrc/ck_gemm_a8w8/gemm_a8w8_common.py`**
```
from dataclasses import dataclass
@dataclass
class kernelInstance:
BLOCK_SIZE: int
```

**`csrc/ck_gemm_a8w8/gemm_a8w8_tune.cu`**
```
using RowwiseKernel = std::function<
torch::Tensor(torch::Tensor &, torch::Tensor &,
torch::Tensor &, torch::Tensor &, torch::Tensor &, std::optional<torch::Tensor>)>;
using RowwiseKernelMap = std::unordered_map<
```

**`csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`**
```
import os
import sys
import ater
import pandas as pd
```
