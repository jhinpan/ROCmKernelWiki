# Diff summary

- **files changed:** 10
- **lines:** +253 / -170
- **kernel-ish files:** 9

## Files (by churn)

- `example/ck_tile/02_layernorm2d/generate.py`  (+93/-67)
- `include/ck_tile/ops/norm_reduce/block/block_norm_reduce.hpp`  (+78/-48)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_default_policy.hpp`  (+30/-27)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_one_pass.hpp`  (+27/-13)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_two_pass.hpp`  (+11/-9)
- `include/ck_tile/ops/norm_reduce/block/block_norm_reduce_problem.hpp`  (+7/-2)
- `include/ck_tile/ops/norm_reduce.hpp`  (+3/-3)
- `example/ck_tile/02_layernorm2d/script/smoke_test.sh`  (+2/-1)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_traits.hpp`  (+2/-0)
- `include/ck_tile/ops/norm_reduce/thread/thread_welford.hpp`  (+0/-0)

## Key added lines (kernel files)

**`example/ck_tile/02_layernorm2d/generate.py`**
```
bool kWelford_,
static constexpr bool kWelford        = kWelford_;
bool kWelford_,
kWelford_,
```

**`include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_default_policy.hpp`**
```
CK_TILE_HOST_DEVICE static constexpr auto GetBlockNormReduce()
using P_ = BlockNormReduceProblem<typename Problem::ComputeDataType,
typename Problem::ComputeDataType,
typename Problem::BlockShape,
```

**`include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_one_pass.hpp`**
```
static constexpr bool kWelford           = Problem::Traits::kWelford;
auto block_norm_reduce      = Policy::template GetBlockNormReduce<Problem>();
auto block_norm_reduce_sync = Policy::template GetBlockNormReduceSync<Problem>();
auto block_norm_reduce_cross_warp_sync =
```

**`include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_two_pass.hpp`**
```
static constexpr bool kWelford           = Problem::Traits::kWelford;
static_assert(kWelford == true, "2 pass only supports welford merge");
auto block_norm_reduce      = Policy::template GetBlockNormReduce<Problem>();
auto block_norm_reduce_sync = Policy::template GetBlockNormReduceSync<Problem>();
```

**`include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_traits.hpp`**
```
bool kWelford_,
static constexpr bool kWelford                         = kWelford_;
```
