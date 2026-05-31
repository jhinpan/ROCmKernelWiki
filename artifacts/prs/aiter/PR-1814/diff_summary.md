# Diff summary

- **files changed:** 14
- **lines:** +558 / -280
- **kernel-ish files:** 13

## Files (by churn)

- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_blockscale.cuh`  (+406/-173)
- `csrc/include/rocm_ops.hpp`  (+42/-40)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4.cuh`  (+21/-19)
- `aiter/fused_moe.py`  (+22/-6)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.cuh`  (+9/-7)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4_bns.cuh`  (+9/-7)
- `aiter/ops/moe_op.py`  (+12/-1)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`  (+7/-6)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`  (+7/-5)
- `aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_qwen3_235b.csv`  (+0/-10)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`  (+7/-3)
- `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`  (+8/-1)
- `aiter/test_common.py`  (+5/-1)
- `csrc/include/moe_ck.h`  (+3/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
@functools.lru_cache(maxsize=2048)
def use_nt(token, topk, e):
use_nt = int(os.environ.get("AITER_USE_NT", "-1"))
if use_nt != -1:
```

**`aiter/ops/moe_op.py`**
```
use_non_temporal_load: bool = False,
splitk: int = 1,
use_non_temporal_load: bool = False,
dst_type: Optional[str] = None,
```

**`aiter/test_common.py`**
```
r["device_time_avg"] = (
r["device_time_sum"] / r["cnt"] if r["cnt"] > 0 else 0
"device_time_avg",
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`**
```
bool nt = false,
hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, o
bool nt = false,
inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, ou
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`**
```
std::optional<int>,
std::optional<bool>)>;
std::optional<int> splitk     = 1,            // splitk
std::optional<bool> nt        = false);
```
