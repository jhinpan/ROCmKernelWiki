# Diff summary

- **files changed:** 31
- **lines:** +623 / -3529
- **kernel-ish files:** 29

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+396/-1092)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload.hpp`  (+0/-1177)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp`  (+0/-823)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp`  (+54/-134)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+55/-92)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+21/-27)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+11/-19)
- `include/ck_tile/ops/reduce/block/block_reduce.hpp`  (+8/-22)
- `include/ck_tile/core/arch/arch.hpp`  (+15/-12)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp`  (+12/-12)
- `example/ck_tile/01_fmha/script/smoke_test_fwd.sh`  (+11/-10)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+5/-12)
- `include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`  (+5/-12)
- `include/ck_tile/core/arch/utility.hpp`  (+0/-15)
- `include/ck_tile/core/numeric/vector_type.hpp`  (+6/-6)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_dropout}, {F_squant}, {F_spad}, {F_skpad}, {F_dp
({F_scheck}) && ({F_skcheck}) && ({F_dcheck}) && ({F_dvcheck}) && ({F_constraint})) {{
using trait_ = fmha_fwd_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}, 
if self.pipeline_tag == 'qr_async':
```

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
<< " GB/s" << std::flush;
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
r.x         = __builtin_amdgcn_readfirstlane(r.x);
r.y         = __builtin_amdgcn_readfirstlane(r.y);
r.z         = __builtin_amdgcn_readfirstlane(r.z);
r.w         = __builtin_amdgcn_readfirstlane(r.w);
```

**`include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`**
```
r.x         = __builtin_amdgcn_readfirstlane(r.x);
r.y         = __builtin_amdgcn_readfirstlane(r.y);
r.z         = __builtin_amdgcn_readfirstlane(r.z);
r.w         = __builtin_amdgcn_readfirstlane(r.w);
```

**`include/ck_tile/core/arch/arch.hpp`**
```
CK_TILE_DEVICE void block_sync_lds()
__builtin_amdgcn_s_waitcnt(0xc07f);
__builtin_amdgcn_s_barrier();
__syncthreads();
```
