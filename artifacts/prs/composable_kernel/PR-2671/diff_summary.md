# Diff summary

- **files changed:** 27
- **lines:** +3759 / -526
- **kernel-ish files:** 25

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+1092/-396)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload.hpp`  (+1177/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp`  (+821/-0)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp`  (+372/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+92/-55)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2_custom_policy.hpp`  (+45/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+19/-11)
- `include/ck_tile/ops/reduce/block/block_reduce.hpp`  (+22/-8)
- `include/ck_tile/core/arch/arch.hpp`  (+12/-15)
- `example/ck_tile/01_fmha/script/smoke_test_fwd.sh`  (+10/-11)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+12/-5)
- `include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`  (+12/-5)
- `include/ck_tile/core/arch/utility.hpp`  (+15/-0)
- `include/ck_tile/core/numeric/vector_type.hpp`  (+6/-6)
- `example/ck_tile/01_fmha/script/benchmark_fwd.sh`  (+0/-11)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
"qr_async_trload" : "ck_tile::BlockFmhaPipelineQRKSVSAsyncTrload",
"qr_async_trload" : "ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC_TRLOAD",
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
from codegen.utils import update_file
{F_trload},
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_dropout}, {F_squant}, {F_spad}, {F_skpad}, {F_dp
const bool has_load_tr = ck_tile::is_load_tr_supported();
```

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
<< " GB/s" << std::flush << std::endl;
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
bool kUseTrLoad_,
static constexpr bool kUseTrLoad                 = kUseTrLoad_;
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
WAVE_NT0   = 0,
WAVE_NT1   = 2,
GROUP_NT0  = 1,
GROUP_NT1  = 3,
```
