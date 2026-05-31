# Diff summary

- **files changed:** 17
- **lines:** +975 / -10
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck_tile/ops/layernorm2d/kernel/layernorm2d_fwd_kernel.hpp`  (+291/-0)
- `example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`  (+191/-0)
- `include/ck_tile/ops/welford/warp/warp_welford.hpp`  (+154/-0)
- `include/ck_tile/ops/welford/thread/thread_welford.hpp`  (+101/-0)
- `include/ck_tile/host/reference/reference_layernorm2d.hpp`  (+69/-0)
- `include/ck_tile/ops/layernorm2d/pipeline/tile_layernorm2d_fwd_shape.hpp`  (+35/-0)
- `example/ck_tile/02_layernorm2d/layernorm2d_fwd.hpp`  (+30/-0)
- `include/ck_tile/ops/layernorm2d/pipeline/block_layernorm2d_fwd_problem.hpp`  (+30/-0)
- `include/ck_tile/host/check_err.hpp`  (+15/-10)
- `example/ck_tile/02_layernorm2d/README.md`  (+22/-0)
- `include/ck_tile/core/numeric/null_type.hpp`  (+13/-0)
- `include/ck_tile/ops/layernorm2d.hpp`  (+9/-0)
- `include/ck_tile/ops/welford.hpp`  (+8/-0)
- `example/ck_tile/02_layernorm2d/CMakeLists.txt`  (+4/-0)
- `example/ck_tile/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`**
```
float layernorm2d_fwd(layernorm2d_fwd_traits t,
layernorm2d_fwd_args a,
const ck_tile::stream_config& s)
if(t.data_type.compare("fp16") == 0)
```

**`example/ck_tile/02_layernorm2d/layernorm2d_fwd.hpp`**
```
struct layernorm2d_fwd_traits
std::string data_type;
struct layernorm2d_fwd_args
const void* p_x;
```

**`include/ck_tile/core/numeric/null_type.hpp`**
```
namespace ck_tile {
struct null_type
} // namespace ck_tile
```

**`include/ck_tile/host/check_err.hpp`**
```
const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
const bool both_infinite_and_same =
std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));
const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
```

**`include/ck_tile/host/reference/reference_layernorm2d.hpp`**
```
namespace ck_tile {
template <typename XDataType,
typename GammaDataType,
typename BetaDataType,
```
