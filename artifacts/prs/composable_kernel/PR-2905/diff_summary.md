# Diff summary

- **files changed:** 11
- **lines:** +441 / -58
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/core/tensor/tile_window.hpp`  (+163/-32)
- `include/ck_tile/core/tensor/load_tile_transpose.hpp`  (+54/-6)
- `include/ck_tile/core/tensor/load_tile.hpp`  (+49/-2)
- `include/ck_tile/core/tensor/store_tile.hpp`  (+51/-0)
- `include/ck_tile/core/tensor/static_distributed_tensor.hpp`  (+37/-4)
- `include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`  (+37/-4)
- `include/ck_tile/core/tensor/tile_distribution.hpp`  (+21/-6)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+15/-0)
- `include/ck_tile/core/container/sequence.hpp`  (+11/-0)
- `include/ck_tile/core/tensor/tile_scatter_gather.hpp`  (+2/-3)
- `include/ck_tile/ops/reduce/block/block_reduce.hpp`  (+1/-1)

## Key added lines (kernel files)

**`include/ck_tile/core/container/sequence.hpp`**
```
template <typename T>
struct is_sequence : std::false_type
template <index_t... Is>
struct is_sequence<sequence<Is...>> : std::true_type
```

**`include/ck_tile/core/tensor/load_tile.hpp`**
```
template <typename TileWindow_,
index_t i_access           = -1,
bool oob_conditional_check = true,
typename                   = std::enable_if_t<std::is_class_v<TileWindow_>>>
```

**`include/ck_tile/core/tensor/load_tile_transpose.hpp`**
```
CK_TILE_DEVICE auto load_tile_transpose_with_offset(
const tile_window_with_static_distribution<BottomTensorView_,
WindowLengths_,
TileDistribution_,
```

**`include/ck_tile/core/tensor/static_distributed_tensor.hpp`**
```
CK_TILE_HOST_DEVICE constexpr auto get_x_indices_from_distributed_indices(
StaticTileDistribution tile_distribution,
DistributedIndices distributed_indices,
decltype(get_partition_index(tile_distribution)) partition_index)
```

**`include/ck_tile/core/tensor/store_tile.hpp`**
```
template <typename BottomTensorView_,
typename WindowLengths_,
typename TileDistribution_,
typename DataType_>
```
