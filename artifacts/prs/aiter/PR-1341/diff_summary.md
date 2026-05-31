# Diff summary

- **files changed:** 14
- **lines:** +2322 / -357
- **kernel-ish files:** 13

## Files (by churn)

- `csrc/ck_tile_gemm_moe_2stages/gen_instances.py`  (+578/-0)
- `op_tests/test_moe_2stage.py`  (+159/-294)
- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages_common.py`  (+448/-0)
- `csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages_common.cuh`  (+328/-0)
- `aiter/fused_moe.py`  (+220/-21)
- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages.cu`  (+218/-0)
- `aiter/ops/moe_op.py`  (+106/-0)
- `csrc/include/rocm_ops.hpp`  (+69/-31)
- `aiter/ops/shuffle.py`  (+86/-0)
- `csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages.h`  (+74/-0)
- `aiter/jit/optCompilerConfig.json`  (+26/-5)
- `csrc/pybind/moe_ck_2stages_pybind.cu`  (+2/-5)
- `csrc/pybind/moe_cktile_2stages_pybind.cu`  (+6/-0)
- `csrc/include/aiter_enum.h`  (+2/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
hidden_pad=0,
intermediate_pad=0,
bias1=None,
bias2=None,
```

**`aiter/ops/moe_op.py`**
```
@compile_ops("module_moe_cktile2stages", fc_name="cktile_moe_gemm1")
def moe_cktile2stages_gemm1_ck(
XQ: Tensor,
WQ: Tensor,
```

**`aiter/ops/shuffle.py`**
```
def shuffle_weight_NK(
x: torch.Tensor, inst_N: int, inst_K: int, use_int4=False
) -> torch.Tensor:
kPerLane = inst_K // (64 // inst_N)
```

**`csrc/ck_tile_gemm_moe_2stages/gen_instances.py`**
```
import os
import argparse
from pathlib import Path
import shutil
```

**`csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages.h`**
```
using MoeKernel        = std::function<torch::Tensor(torch::Tensor&,
torch::Tensor&,
torch::Tensor&,
torch::Tensor&,
```
