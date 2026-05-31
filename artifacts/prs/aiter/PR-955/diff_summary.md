# Diff summary

- **files changed:** 19
- **lines:** +1520 / -605
- **kernel-ish files:** 14

## Files (by churn)

- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_common.py`  (+31/-472)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gen_instances.py`  (+305/-0)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.py`  (+255/-0)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/include/gemm_a8w8_blockscale_bpreshuffle_common.cuh`  (+166/-0)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle.cu`  (+120/-0)
- `aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv`  (+118/-0)
- `aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv`  (+118/-0)
- `aiter/configs/a8w8_blockscale_untuned_gemm.csv`  (+1/-117)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_common.py`  (+112/-0)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.cu`  (+84/-0)
- `op_tests/test_gemm_a8w8_blockscale.py`  (+37/-14)
- `aiter/ops/gemm_op_a8w8.py`  (+43/-0)
- `aiter/jit/optCompilerConfig.json`  (+30/-0)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/README.md`  (+28/-0)
- `csrc/include/rocm_ops.hpp`  (+22/-0)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a8w8.py`**
```
@compile_ops(
"module_gemm_a8w8_blockscale_bpreshuffle",
fc_name="gemm_a8w8_blockscale_bpreshuffle",
gen_fake=gen_gemm_a8w8_blockscale_ck_fake_tensors,
```

**`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_common.py`**
```
0:   kernelInstance(256,     1,   128,   128,   128,   128,   128,  16,  16,  32,   32,    2,    2,     [ 8, 32, 1],    
1:   kernelInstance(256,     1,   128,   128,   128,    64,   128,  16,  16,  32,   32,    2,    1,     [ 8, 32, 1],    
2:   kernelInstance(256,     1,   128,   128,    64,   128,   128,  16,  16,  32,   32,    1,    2,     [ 8, 32, 1],    
3:   kernelInstance(256,     1,   128,   128,    64,    64,   128,  16,  16,  32,   32,    1,    1,     [ 8, 32, 1],    
```

**`csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle.cu`**
```
using BlockwiseKernel = std::function<torch::Tensor(
torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;
struct IntTupleHash
size_t operator()(const std::tuple<int, int, int>& t) const
```

**`csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_common.py`**
```
from dataclasses import dataclass
@dataclass
class kernelInstance:
BLOCK_SIZE: int
```

**`csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.cu`**
```
using BlockwiseKernel = std::function<torch::Tensor(
torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;
using BlockwiseKernelMap = std::unordered_map<int, BlockwiseKernel>;
static constexpr int nextPow2(unsigned int num)
```
