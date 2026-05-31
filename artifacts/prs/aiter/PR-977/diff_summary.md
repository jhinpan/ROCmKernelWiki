# Diff summary

- **files changed:** 15
- **lines:** +2241 / -100
- **kernel-ish files:** 13

## Files (by churn)

- `csrc/ck_tile_gemm_moe_2stages/gen_instances.py`  (+554/-0)
- `op_tests/test_moe_2stage.py`  (+450/-74)
- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages_common.py`  (+351/-0)
- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages_common.cuh`  (+327/-0)
- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages.cu`  (+180/-0)
- `aiter/ops/moe_op.py`  (+73/-0)
- `aiter/test_common.py`  (+59/-12)
- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages.h`  (+68/-0)
- `aiter/ops/shuffle.py`  (+67/-0)
- `aiter/fused_moe.py`  (+43/-13)
- `csrc/include/rocm_ops.hpp`  (+39/-0)
- `aiter/jit/optCompilerConfig.json`  (+18/-0)
- `csrc/pybind/moe_cktile_2stages_pybind.cu`  (+9/-0)
- `3rdparty/composable_kernel`  (+1/-1)
- `csrc/rocm_ops.cpp`  (+2/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
if M * topk <= num_experts:
max_num_tokens_padded = M * topk * block_size
def swiglu(x_glu, x_linear, alpha: float = 1.702, limit: float = 7.0):
x_glu = x_glu.clamp(min=None, max=limit)
```

**`aiter/ops/moe_op.py`**
```
@compile_ops("module_moe_cktile2stages",  fc_name="cktile_moe_gemm1")
def moe_cktile2stages_gemm1_ck(
XQ                  : Tensor,
WQ                  : Tensor,
```

**`aiter/ops/shuffle.py`**
```
def shuffle_weight_NK(x: torch.Tensor, inst_N: int, inst_K: int, use_int4=False) -> torch.Tensor:
kPerLane = inst_K // (64 //  inst_N)
if(use_int4):
kPerLane *= 2
```

**`aiter/test_common.py`**
```
torch.cuda.synchronize()
profile_memory=False,
with_stack=False,
tpf.tensorboard_trace_handler(f"./aiter_logs/gpu_id_{gpu_id}")
```

**`csrc/ck_tile_gemm_moe_2stages/gen_instances.py`**
```
import os
import argparse
from pathlib import Path
import shutil
```
