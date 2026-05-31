# Diff summary

- **files changed:** 10
- **lines:** +775 / -74
- **kernel-ish files:** 9

## Files (by churn)

- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4.cuh`  (+367/-0)
- `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`  (+192/-14)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`  (+139/-22)
- `aiter/fused_moe.py`  (+35/-17)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`  (+13/-5)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`  (+10/-8)
- `op_tests/test_moe_2stage.py`  (+10/-6)
- `aiter/jit/optCompilerConfig.json`  (+7/-1)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.cuh`  (+1/-1)
- `aiter/ops/moe_op.py`  (+1/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
from aiter import get_hip_quant as get_quant
from aiter.utility.fp4_utils import moe_mxfp4_sort
run_1stage = quant_type in [QuantType.per_128x128]
if "ck" in kernelName1 or q_dtype_w in [
```

**`aiter/ops/moe_op.py`**
```
torch.uint8: "fp4x2",
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`**
```
MoeKernel moe_dispatch(std::string &kernelName, int block_m, int inter_dim)
return moe_stage2_heuristic_dispatch(block_m, inter_dim);
if (hidden_states.dtype() == at::ScalarType::Byte && w1.dtype() == at::ScalarType::Byte)
auto kernel = moe_dispatch<1>(kernelName, MPerBlock, N);
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`**
```
using S     = ck::Sequence<Is...>;
using I4    = ck::pk_i4_t;
using I8    = int8_t;
using I32   = int;
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.cuh`**
```
2,    CShuffleNXDLPerWave,   S<1, CShuffleMLane, 1, CShuffleNLane>, S<EVec, D0Vec, D1Vec, D2Vec>,
```
