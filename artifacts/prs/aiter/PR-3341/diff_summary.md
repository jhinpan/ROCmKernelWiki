# Diff summary

- **files changed:** 44
- **lines:** +773 / -311
- **kernel-ish files:** 8

## Files (by churn)

- `aiter/fused_moe_asmjit_aot.py`  (+360/-0)
- `aiter/fused_moe_ptpc_fp8.py`  (+0/-256)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+234/-5)
- `op_tests/test_moe.py`  (+68/-3)
- `aiter/fused_moe.py`  (+22/-34)
- `aiter/configs/tuned_fmoe.csv`  (+41/-10)
- `csrc/cpp_itfs/hsaco_tools.py`  (+21/-1)
- `aiter/configs/model_configs/qwen3_5_397b_fp8_ptpc_untuned_fmoe.csv`  (+19/-0)
- `aiter/utility/base_tuner.py`  (+6/-0)
- `op_tests/test_moe_asmjit_aot.py`  (+2/-2)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=10-K=128-N=4096-with_silu=False-BLOCK_TILE_SIZE_M=128-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=False.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=10-K=128-N=4096-with_silu=False-BLOCK_TILE_SIZE_M=128-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=True.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=10-K=128-N=4096-with_silu=False-BLOCK_TILE_SIZE_M=64-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=False.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=10-K=128-N=4096-with_silu=False-BLOCK_TILE_SIZE_M=64-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=True.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=11-K=128-N=4096-with_silu=False-BLOCK_TILE_SIZE_M=128-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=False.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
if metadata.stage0 is not None:
return metadata.stage0(
hidden_states,
topk_weight,
```

**`aiter/fused_moe_asmjit_aot.py`**
```
from typing import Any, Optional
import torch
import aiter
from aiter import ActivationType, QuantType
```

**`aiter/utility/base_tuner.py`**
```
self.parser.add_argument(
"--e2e_tune",
action="store_true",
required=False,
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`**
```
from aiter.fused_moe_asmjit_aot import fused_moe_asmjit_aot
from aiter.fused_moe_asmjit_aot import get_tune_space
def run_config(self, args, target_fused_moe=None, try_extra_ref=False):
if target_fused_moe is None:
```

**`csrc/cpp_itfs/hsaco_tools.py`**
```
from aiter.jit.utils.chip_info import get_gfx
from csrc.cpp_itfs.utils import AITER_CORE_DIR
return CallableKernel
class HSACO:
```
