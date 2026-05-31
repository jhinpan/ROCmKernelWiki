# Diff summary

- **files changed:** 13
- **lines:** +731 / -185
- **kernel-ish files:** 11

## Files (by churn)

- `example/ck_tile/10_rmsnorm2d/generate.py`  (+169/-88)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_model_sensitive_pass.hpp`  (+228/-0)
- `include/ck_tile/ops/reduce/block/block_reduce2d.hpp`  (+133/-0)
- `example/ck_tile/10_rmsnorm2d/script/perf_test.sh`  (+70/-33)
- `example/ck_tile/10_rmsnorm2d/script/smoke_test.sh`  (+30/-24)
- `example/ck_tile/10_rmsnorm2d/example_rmsnorm2d_fwd.cpp`  (+31/-8)
- `example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.cpp`  (+23/-14)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_traits.hpp`  (+24/-7)
- `include/ck_tile/ops/rmsnorm2d/kernel/rmsnorm2d_fwd_kernel.hpp`  (+10/-7)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_default_policy.hpp`  (+9/-0)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_one_pass.hpp`  (+1/-4)
- `example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.hpp`  (+2/-0)
- `include/ck_tile/ops/rmsnorm2d.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/10_rmsnorm2d/example_rmsnorm2d_fwd.cpp`**
```
.insert("repeat", "1", "hot iter")
.insert("s", "0", "sensitive model mode, 0: for no specific model, 1: for T5-like model");
template <typename DataType, int USEModelSensitive>
ck_tile::Rmsnorm2dFusedAddEnum::NO_ADD,     // fuse add
```

**`example/ck_tile/10_rmsnorm2d/generate.py`**
```
ck_tile::index_t kFusedQuant_ = 0,
ck_tile::index_t kUseModelSensitiveRMSNorm_ = 0>
static constexpr ck_tile::index_t kFusedAdd                 = kFusedAdd_;
static constexpr ck_tile::index_t kFusedQuant               = kFusedQuant_;
```

**`example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.cpp`**
```
.insert("repeat", "20", "hot iter")
.insert("s", "0", "sensitive model mode, 0: for no specific model, 1: for T5-like model");
ck_tile::index_t m                    = arg_parser.get_int("m");
ck_tile::index_t n                    = arg_parser.get_int("n");
```

**`example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.hpp`**
```
int use_model_sensitive_rmsnorm = 0; // 0: Use default RMSNorm; 1: Use T5-like implementation
```

**`include/ck_tile/ops/reduce/block/block_reduce2d.hpp`**
```
template <typename Problem_, typename Policy_ = void>
struct BlockReduce2dTreeCrossWarpSync
using Problem    = remove_cvref_t<Problem_>;
using BlockShape = typename Problem::BlockShape;
```
