# Diff summary

- **files changed:** 14
- **lines:** +675 / -127
- **kernel-ish files:** 13

## Files (by churn)

- `aiter/utility/mp_tuner.py`  (+399/-87)
- `aiter/utility/base_tuner.py`  (+60/-10)
- `gradlib/README.md`  (+67/-0)
- `hsa/gfx942/fmoe_2stages/tune.py`  (+17/-7)
- `gradlib/gradlib/GemmTuner.py`  (+20/-3)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+19/-3)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`  (+15/-3)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py`  (+15/-3)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`  (+15/-3)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`  (+15/-3)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.py`  (+13/-2)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`  (+10/-1)
- `hsa/gfx950/fmoe_2stages/tune.py`  (+7/-1)
- `aiter/jit/core.py`  (+3/-1)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
if "cu_num" not in keys:
keys.append("cu_num")
```

**`aiter/utility/base_tuner.py`**
```
"timeout": None,  # 100s timeout for per test
"warmup": 5,  # 5 warmup iters for profiling
"iters": 101,  # 101 run iters for profiling
INVALID_TIME = -1  # op not support or error
```

**`aiter/utility/mp_tuner.py`**
```
from multiprocessing import TimeoutError as MPTimeoutError
from aiter import logger
device = torch.device(f"cuda:{gpu_id}")
torch.cuda.set_device(device)
```

**`csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`**
```
"num_warmup": args.warmup,
"num_iters": args.iters,
ret = mp_tuner(
tasks_data,
```

**`csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py`**
```
"num_warmup": args.warmup,
"num_iters": args.iters,
ret = mp_tuner(
tasks_data,
```
