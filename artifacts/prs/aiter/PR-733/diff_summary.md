# Diff summary

- **files changed:** 8
- **lines:** +306 / -138
- **kernel-ish files:** 8

## Files (by churn)

- `aiter/utility/mp_tuner.py`  (+79/-13)
- `gradlib/gradlib/GemmTuner.py`  (+59/-18)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`  (+42/-32)
- `aiter/test_common.py`  (+59/-12)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+28/-41)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`  (+16/-11)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`  (+16/-11)
- `hsa/gfx942/fmoe_2stages/tune.py`  (+7/-0)

## Key added lines (kernel files)

**`aiter/test_common.py`**
```
torch.cuda.synchronize()
profile_memory=False,
with_stack=False,
tpf.tensorboard_trace_handler(f"./aiter_logs/gpu_id_{gpu_id}")
```

**`aiter/utility/mp_tuner.py`**
```
gpuID = torch.cuda.current_device()
except RuntimeError as e:
print(f"run gpu func error: info:{info}\t {e}")
if us == 0:
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`**
```
kernelName,
out_reset = torch.zeros(
out.shape[0], out.shape[1], dtype=dtype, device=torch.cuda.current_device()
def generate_data(m, n, k, seed, device="cuda", dtype=dtypes.bf16):
```

**`csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`**
```
def generate_data(m, n, k, seed, device="cuda"):
torch.manual_seed(seed)
x = torch.randint(-20, 20, (m, k), dtype=dtypes.i8, device=device)
weight = torch.randint(-20, 20, (n, k), dtype=dtypes.i8, device=device)
```

**`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`**
```
def generate_data(m, n, k, seed, device="cuda"):
torch.manual_seed(seed)
x = (torch.rand((m, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
weight = (torch.rand((n, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
```
