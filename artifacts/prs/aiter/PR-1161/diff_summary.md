# Diff summary

- **files changed:** 13 (diff was byte-capped; summary is partial)
- **lines:** +1502 / -1128
- **kernel-ish files:** 8

## Files (by churn)

- `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`  (+702/-666)
- `aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv`  (+318/-281)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+203/-55)
- `csrc/py_itfs_cu/asm_gemm_a8w8.cu`  (+166/-47)
- `aiter/ops/gemm_op_a8w8.py`  (+39/-34)
- `aiter/configs/asm_a8w8_gemm.csv`  (+25/-11)
- `csrc/include/asm_gemm_a8w8.h`  (+9/-12)
- `csrc/ck_gemm_a8w8_bpreshuffle/gen_instances.py`  (+13/-2)
- `csrc/ck_gemm_a8w8_bpreshuffle/README.md`  (+7/-6)
- `aiter/utility/base_tuner.py`  (+9/-3)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.cu`  (+7/-4)
- `csrc/include/rocm_ops.hpp`  (+3/-6)
- `aiter/jit/optCompilerConfig.json`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a8w8.py`**
```
kernelName: str,
bias: Optional[Tensor],  # bias:[1, N] f32
bpreshuffle: Optional[bool] = True,
splitK: Optional[int] = None,
```

**`aiter/utility/base_tuner.py`**
```
tunedf_subset = tunedf[self.untunedf.columns].astype(self.untunedf.dtypes)
tunedf_subset.apply(tuple, axis=1)
print(self.untunedf)
cu_num, m, n, k, *rest = info[0]
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.cu`**
```
if (Y.dtype() == at::ScalarType::BFloat16)
blockwise_dispatch<F32, B16>(kernelId)(XQ, WQ, x_scale, w_scale, Y);
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`**
```
from aiter.jit.core import get_asm_dir
def run_gemm_a8w8_asm(
kernelName,
dtype=dtypes.bf16,
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gen_instances.py`**
```
import sys
os.path.join(self.instances_path, f"{k.name}_dFP32_eBF16.cpp")
).write_text(INSTANCE_dFP32_eBF16)
aiter_dir = os.path.dirname(
```
