# Diff summary

- **files changed:** 28 (diff was byte-capped; summary is partial)
- **lines:** +4526 / -167
- **kernel-ish files:** 23

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`  (+1338/-0)
- `example/ck_tile/01_fmha/fmha_bwd.cpp`  (+932/-0)
- `example/ck_tile/01_fmha/generate.py`  (+663/-39)
- `example/ck_tile/01_fmha/fmha_bwd.hpp`  (+359/-0)
- `include/ck_tile/ops/fmha/block/block_dropout.hpp`  (+329/-0)
- `include/ck_tile/core/arch/generic_memory_space_atomic.hpp`  (+175/-0)
- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+88/-47)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+72/-42)
- `include/ck_tile/core/utility/philox_rand.hpp`  (+89/-0)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+71/-3)
- `include/ck_tile/ops/fmha/block/block_masking.hpp`  (+66/-6)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+60/-0)
- `include/ck_tile/core/tensor/update_tile.hpp`  (+55/-0)
- `example/ck_tile/01_fmha/CMakeLists.txt`  (+35/-4)
- `include/ck_tile/host/host_tensor.hpp`  (+32/-4)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/fmha_bwd.cpp`**
```
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
using size_type = typename std::vector<T>::size_type;
os << "[";
```

**`example/ck_tile/01_fmha/fmha_bwd.hpp`**
```
template <typename DataType>
struct FmhaBwdTypeConfig;
template <>
struct FmhaBwdTypeConfig<ck_tile::half_t>
```

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
.insert("p_drop", "0", "0~1 probability of dropout")
.insert("drop_seed", "1", "seed for random number generator")
.insert("drop_offset", "0", "offset for random number generator")
auto get_elimit<ck_tile::bf16_t>(std::string /*init_method*/)
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
using QDataType             = ck_tile::half_t;
using KDataType             = ck_tile::half_t;
using VDataType             = ck_tile::half_t;
using BiasDataType          = ck_tile::half_t;
```

**`example/ck_tile/01_fmha/generate.py`**
```
{F_dropout},
{F_squant},
typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::RandValOutputDataType,
{F_pipeline_enum}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_dropout}, {F_squant}, {F_spad}, {F_skpad}, {F_dpad}, {F_dvpa
```
