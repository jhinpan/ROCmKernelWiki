# Diff summary

- **files changed:** 10
- **lines:** +63 / -26
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/common/generic_2d_block_shape.hpp`  (+34/-17)
- `include/ck_tile/ops/add_rmsnorm2d_rdquant/kernel/add_rmsnorm2d_rdquant_fwd_kernel.hpp`  (+5/-1)
- `include/ck_tile/ops/layernorm2d/kernel/layernorm2d_fwd_kernel.hpp`  (+5/-1)
- `include/ck_tile/ops/rmsnorm2d/kernel/rmsnorm2d_fwd_kernel.hpp`  (+5/-1)
- `include/ck_tile/ops/smoothquant/kernel/moe_smoothquant_kernel.hpp`  (+5/-1)
- `include/ck_tile/ops/smoothquant/kernel/smoothquant_kernel.hpp`  (+5/-1)
- `test/ck_tile/add_rmsnorm2d_rdquant/instances/add_rmsnorm2d_rdquant_fwd_instance_common.hpp`  (+1/-1)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_instance_common.hpp`  (+1/-1)
- `test/ck_tile/rmsnorm2d/generate.py`  (+1/-1)
- `test/ck_tile/smoothquant/instances/smoothquant_instance_common.hpp`  (+1/-1)

## Key added lines (kernel files)

**`include/ck_tile/ops/add_rmsnorm2d_rdquant/kernel/add_rmsnorm2d_rdquant_fwd_kernel.hpp`**
```
CK_TILE_HOST static constexpr auto BlockSize()
return is_wave32() ? Problem::BlockShape::template GetBlockSize<true>()
: Problem::BlockShape::template GetBlockSize<false>();
```

**`include/ck_tile/ops/common/generic_2d_block_shape.hpp`**
```
template <bool isHostWave32>
static constexpr index_t GetWarpPerBlock_M()
constexpr index_t warp_size    = isHostWave32 ? 32 : get_warp_size();
constexpr bool is_warp_per_row = ThreadPerBlock_N <= warp_size;
```

**`include/ck_tile/ops/layernorm2d/kernel/layernorm2d_fwd_kernel.hpp`**
```
CK_TILE_HOST static constexpr auto BlockSize()
return is_wave32() ? Problem::BlockShape::template GetBlockSize<true>()
: Problem::BlockShape::template GetBlockSize<false>();
```

**`include/ck_tile/ops/rmsnorm2d/kernel/rmsnorm2d_fwd_kernel.hpp`**
```
CK_TILE_HOST static constexpr auto BlockSize()
return is_wave32() ? Problem::BlockShape::template GetBlockSize<true>()
: Problem::BlockShape::template GetBlockSize<false>();
```

**`include/ck_tile/ops/smoothquant/kernel/moe_smoothquant_kernel.hpp`**
```
CK_TILE_HOST static constexpr auto BlockSize()
return is_wave32() ? Problem::BlockShape::template GetBlockSize<true>()
: Problem::BlockShape::template GetBlockSize<false>();
```
