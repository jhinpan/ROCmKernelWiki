# Diff summary

- **files changed:** 14
- **lines:** +1158 / -282
- **kernel-ish files:** 6

## Files (by churn)

- `hsa/gfx950/fmoe_2stages/tune.py`  (+0/-191)
- `csrc/ck_gemm_moe_2stages_codegen/README.md`  (+161/-0)
- `csrc/ck_gemm_a8w8_bpreshuffle/README.md`  (+148/-6)
- `csrc/ck_gemm_a4w4_blockscale/README.md`  (+134/-8)
- `csrc/ck_batched_gemm_a8w8/README.md`  (+134/-6)
- `csrc/ck_gemm_a8w8/README.md`  (+131/-9)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/README.md`  (+131/-8)
- `csrc/ck_batched_gemm_bf16/README.md`  (+131/-7)
- `csrc/ck_gemm_a8w8_blockscale/README.md`  (+130/-8)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+37/-17)
- `aiter/utility/mp_tuner.py`  (+15/-10)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+4/-4)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`  (+1/-4)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py`  (+1/-4)

## Key added lines (kernel files)

**`aiter/utility/mp_tuner.py`**
```
print(f"run gpu func warning: info:{info}\t {e}", flush=True)
if printLog:
print(f"GPU Runtime Error in process:{pid} info:{info}: {e}")
if printLog:
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`**
```
kernel.MTile,
kernel.NTile,
kernel.KTile,
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`**
```
if get_gfx() == "gfx950":
extraInfo_1stage = ""
if q_dtype_a == dtypes.i8:
quantDtype = "Int8"
```
