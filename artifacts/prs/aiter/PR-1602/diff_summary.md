# Diff summary

- **files changed:** 93
- **lines:** +97 / -81
- **kernel-ish files:** 93

## Files (by churn)

- `op_tests/op_benchmarks/triton/bench_deepgemm_attention.py`  (+4/-11)
- `op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_a16w16.py`  (+7/-3)
- `op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_a16w16.py`  (+7/-3)
- `op_tests/triton_tests/quant/test_fused_mxfp4_quant.py`  (+6/-4)
- `op_tests/triton_tests/rope/test_rope.py`  (+6/-4)
- `op_tests/op_benchmarks/triton/bench_la_paged_decode.py`  (+1/-6)
- `op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16.py`  (+5/-2)
- `op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16_fused.py`  (+5/-2)
- `op_tests/triton_tests/test_activation.py`  (+5/-2)
- `op_tests/triton_tests/test_fused_kv_cache.py`  (+2/-5)
- `op_tests/triton_tests/gemm/fused/test_fused_gemm_afp4wfp4_mul_add.py`  (+4/-2)
- `op_tests/op_benchmarks/triton/bench_mla_decode_rope.py`  (+4/-1)
- `op_tests/triton_tests/moe/test_moe_mx.py`  (+4/-1)
- `op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`  (+3/-1)
- `op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py`  (+3/-1)

## Key added lines (kernel files)

**`op_tests/op_benchmarks/triton/bench_batched_gemm_a16w16.py`**
```
from op_tests.triton_tests.gemm.batched.test_batched_gemm_bf16 import (
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_a16wfp4.py`**
```
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a16wfp4 import (
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py`**
```
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a8w8 import (
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py`**
```
from op_tests.triton_tests.gemm.batched.test_batched_gemm_afp4wfp4 import (
```

**`op_tests/op_benchmarks/triton/bench_deepgemm_attention.py`**
```
from aiter.test_common import run_perftest
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
```
