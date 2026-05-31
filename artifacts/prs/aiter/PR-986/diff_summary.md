# Diff summary

- **files changed:** 78
- **lines:** +378 / -88
- **kernel-ish files:** 3

## Files (by churn)

- `csrc/py_itfs_cu/asm_fmoe.cu`  (+320/-40)
- `op_tests/test_mla_fp8.py`  (+33/-27)
- `op_tests/test_moe_ep.py`  (+25/-21)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_gelu_1tg_32x128.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_gelu_1tg_32x192.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_gelu_1tg_32x256.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_gelu_1tg_32x320.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_gelu_1tg_32x384.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_gelu_1tg_32x448.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_gelu_1tg_32x512.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_tkw1_gelu_1tg_32x128.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_tkw1_gelu_1tg_32x192.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_tkw1_gelu_1tg_32x256.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_tkw1_gelu_1tg_32x320.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_tkw1_gelu_1tg_32x384.co`  (+0/-0)

## Key added lines (kernel files)

**`csrc/py_itfs_cu/asm_fmoe.cu`**
```
static std::unordered_map<int, FMoeKernelConfig> multix_kernel_int8_vs_configs = {
{"_ZN5aiter53fmoe_bf16_pertokenInt8_g1u1_vs_multix_silu_1tg_32x384E",
"fmoe/silu/fmoe_bf16_pertokenInt8_g1u1_vs_multix_silu_1tg_32x384.co",
{"_ZN5aiter53fmoe_bf16_pertokenInt8_g1u1_vs_multix_silu_1tg_32x512E",
```

**`op_tests/test_mla_fp8.py`**
```
def cal_diff(
x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))
```

**`op_tests/test_moe_ep.py`**
```
from aiter.test_common import checkAllclose, run_perftest, perftest, benchmark
import pandas as pd
@benchmark()
return {"us": avg_b}
```
