# Diff summary

- **files changed:** 16
- **lines:** +228 / -178
- **kernel-ish files:** 15

## Files (by churn)

- `aiter/jit/core.py`  (+157/-118)
- `aiter/ops/gemm_op_a8w8.py`  (+17/-11)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile_common.py`  (+11/-11)
- `aiter/jit/optCompilerConfig.json`  (+8/-8)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gen_instances.py`  (+6/-6)
- `aiter/ops/batched_gemm_op_a8w8.py`  (+7/-4)
- `aiter/tuned_gemm.py`  (+5/-5)
- `aiter/ops/gemm_op_a4w4.py`  (+5/-3)
- `aiter/fused_moe.py`  (+3/-3)
- `aiter/ops/batched_gemm_op_bf16.py`  (+3/-3)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile.cu`  (+1/-1)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile_tune.cu`  (+1/-1)
- `csrc/cktile_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle_cktile.h`  (+1/-1)
- `csrc/pybind/gemm_a8w8_bpreshuffle_cktile_pybind.cu`  (+1/-1)
- `csrc/pybind/gemm_a8w8_bpreshuffle_cktile_tune_pybind.cu`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
AITER_CONFIGS,
config_path = os.path.dirname(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
tune_file = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
```

**`aiter/jit/core.py`**
```
f"{AITER_ROOT_DIR}/aiter/configs/bf16_tuned_gemm.csv",
class AITER_CONFIG(object):
@property
def AITER_CONFIG_GEMM_A4W4_FILE(self):
```

**`aiter/ops/batched_gemm_op_a8w8.py`**
```
AITER_CONFIGS,
"Loading CKBatchedGEMM config from:",
AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE,
AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE
```

**`aiter/ops/batched_gemm_op_bf16.py`**
```
AITER_CONFIGS,
AITER_CONFIGS.AITER_CONFIG_BF16_BATCHED_GEMM_FILE
f"shape is B:{B}, M:{M}, N:{N}, K:{K} dtype is bf16, is tuned on cu_num = {cu_num} in {AITER_CONFIGS.AITER_CONFIG_BF16_B
```

**`aiter/ops/gemm_op_a4w4.py`**
```
AITER_CONFIGS,
gemm_dict = pd.read_csv(
AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE
).drop_duplicates()
```
