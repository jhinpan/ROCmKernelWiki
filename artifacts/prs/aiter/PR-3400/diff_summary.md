# Diff summary

- **files changed:** 27
- **lines:** +357 / -0
- **kernel-ish files:** 1

## Files (by churn)

- `op_tests/test_fused_moe_ptpc_acc.py`  (+288/-0)
- `aiter/configs/model_configs/qwen3_5_122b_fp8_ptpc_untuned_fmoe.csv`  (+19/-0)
- `aiter/configs/model_configs/qwen3_5_35b_fp8_ptpc_untuned_fmoe.csv`  (+19/-0)
- `aiter/configs/model_configs/qwen3_5_122b_fp8_ptpc_tuned_fmoe.csv`  (+16/-0)
- `aiter/configs/model_configs/qwen3_5_35b_fp8_ptpc_tuned_fmoe.csv`  (+15/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=9-K=128-N=2048-with_silu=False-BLOCK_TILE_SIZE_M=128-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=False.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=9-K=128-N=2048-with_silu=False-BLOCK_TILE_SIZE_M=128-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=True.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=9-K=128-N=2048-with_silu=False-BLOCK_TILE_SIZE_M=64-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=False.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=9-K=128-N=2048-with_silu=False-BLOCK_TILE_SIZE_M=64-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=True.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=9-K=128-N=3072-with_silu=False-BLOCK_TILE_SIZE_M=128-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=False.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=9-K=128-N=3072-with_silu=False-BLOCK_TILE_SIZE_M=128-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=True.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=9-K=128-N=3072-with_silu=False-BLOCK_TILE_SIZE_M=64-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=False.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down-weight_dtype=torch.float8_e4m3fnuz-TOPK=9-K=128-N=3072-with_silu=False-BLOCK_TILE_SIZE_M=64-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Token-dyn=True.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down_loopn-weight_dtype=torch.float8_e4m3fnuz-TOPK=9-K=128-N=2048-BLOCK_TILE_SIZE_M=16-BLOCK_TILE_SIZE_N=16-fp8_ptpc=True-BLOCK_N=1024-atomic_write=False-STAGES=3.co`  (+0/-0)
- `hsa/gfx942/fmoe_asmjit/moe_2stage_down_loopn-weight_dtype=torch.float8_e4m3fnuz-TOPK=9-K=128-N=3072-BLOCK_TILE_SIZE_M=16-BLOCK_TILE_SIZE_N=16-fp8_ptpc=True-BLOCK_N=1024-atomic_write=False-STAGES=3.co`  (+0/-0)

## Key added lines (kernel files)

**`op_tests/test_fused_moe_ptpc_acc.py`**
```
"""``fused_moe`` FP8 per-token (PTPC) accuracy vs torch MoE reference."""
from __future__ import annotations
import argparse
import sys
```
