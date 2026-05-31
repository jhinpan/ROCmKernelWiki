# Diff summary

- **files changed:** 16
- **lines:** +1992 / -89
- **kernel-ish files:** 14

## Files (by churn)

- `op_tests/tuning_tests/test_run_config.py`  (+499/-0)
- `op_tests/tuning_tests/test_tune_pipeline.py`  (+416/-0)
- `op_tests/tuning_tests/test_tuner_infra.py`  (+328/-0)
- `op_tests/tuning_tests/test_mp_tuner_logic.py`  (+295/-0)
- `aiter/utility/base_tuner.py`  (+138/-72)
- `op_tests/tuning_tests/test_csv_validation.py`  (+149/-0)
- `op_tests/tuning_tests/README.md`  (+87/-0)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`  (+22/-2)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+17/-3)
- `aiter/configs/tuned_fmoe.csv`  (+7/-6)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+8/-2)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`  (+7/-1)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`  (+7/-1)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`  (+5/-1)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py`  (+5/-1)

## Key added lines (kernel files)

**`aiter/utility/base_tuner.py`**
```
if status.startswith("mismatch"):
detail = status[len("mismatch") :].lstrip(":").strip()
return "MISMATCH", detail or "output mismatch vs reference"
def _emit_repro_csv(self, failed_repros, report_file=None):
```

**`csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`**
```
status = (
if err_ratio <= args.errRatio
else f"mismatch:err_ratio={err_ratio:.4f}(>{args.errRatio})"
```

**`csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py`**
```
status = (
if err_ratio <= args.errRatio
else f"mismatch:err_ratio={err_ratio:.4f}(>{args.errRatio})"
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`**
```
status = (
if err_ratio <= args.errRatio
else f"mismatch:err_ratio={err_ratio:.4f}(>{args.errRatio})"
torch.cuda.empty_cache()
```

**`csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`**
```
status = (
if err_ratio <= args.errRatio
else f"mismatch:err_ratio={err_ratio:.4f}(>{args.errRatio})"
torch.cuda.empty_cache()
```
