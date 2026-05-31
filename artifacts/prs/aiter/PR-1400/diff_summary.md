# Diff summary

- **files changed:** 17
- **lines:** +2100 / -15
- **kernel-ish files:** 13

## Files (by churn)

- `csrc/cktile_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle_cktile_common.cuh`  (+386/-0)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile_common.py`  (+384/-0)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gen_instances.py`  (+281/-0)
- `aiter/configs/a8w8_bpreshuffle_cktile_tuned_gemm.csv`  (+221/-0)
- `aiter/configs/a8w8_bpreshuffle_cktile_untuned_gemm.csv`  (+221/-0)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile_tune.py`  (+177/-0)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile.cu`  (+125/-0)
- `aiter/ops/gemm_op_a8w8.py`  (+96/-15)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile_tune.cu`  (+88/-0)
- `aiter/jit/optCompilerConfig.json`  (+37/-0)
- `csrc/include/rocm_ops.hpp`  (+23/-0)
- `csrc/cktile_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle_cktile.h`  (+20/-0)
- `csrc/cktile_gemm_a8w8_bpreshuffle/README.md`  (+18/-0)
- `aiter/jit/core.py`  (+11/-0)
- `csrc/pybind/gemm_a8w8_bpreshuffle_cktile_tune_pybind.cu`  (+6/-0)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_CKTILE = os.getenv(
"AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_CKTILE",
f"{AITER_ROOT_DIR}/aiter/configs/a8w8_bpreshuffle_cktile_tuned_gemm.csv",
AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_CKTILE_FILE = get_config_file(
```

**`aiter/ops/gemm_op_a8w8.py`**
```
AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_CKTILE_FILE,
def gen_gemm_a8w8_bpreshuffle_cktile_fake_tensors(
XQ: torch.Tensor,
WQ: torch.Tensor,
```

**`csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile.cu`**
```
using RowwiseKernel = std::function<torch::Tensor(
torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;
struct IntTupleHash
size_t operator()(const std::tuple<int, int, int>& t) const
```

**`csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile_common.py`**
```
from dataclasses import dataclass
import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
```

**`csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile_tune.cu`**
```
using RowwiseKernel = std::function<torch::Tensor(
torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;
using RowwiseKernelMap = std::unordered_map<int, RowwiseKernel>;
static constexpr int nextPow2(unsigned int num)
```
