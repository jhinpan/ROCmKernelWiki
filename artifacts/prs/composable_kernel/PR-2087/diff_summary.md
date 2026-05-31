# Diff summary

- **files changed:** 12
- **lines:** +445 / -48
- **kernel-ish files:** 10

## Files (by churn)

- `example/ck_tile/99_toy_example/00_add_vector/add_vector.cpp`  (+166/-0)
- `example/ck_tile/99_toy_example/00_add_vector/add_vector.hpp`  (+140/-0)
- `example/ck_tile/99_toy_example/01_add/add.hpp`  (+56/-30)
- `example/ck_tile/99_toy_example/00_add_vector/reference_add_vector.hpp`  (+31/-0)
- `example/ck_tile/99_toy_example/00_add_vector/CMakeLists.txt`  (+22/-0)
- `example/ck_tile/99_toy_example/01_add/add.cpp`  (+11/-6)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+6/-3)
- `example/ck_tile/99_toy_example/03_flash_attention_fwd/block_gemm_pipeline_agmem_bgmem_creg_v2_askiplds.hpp`  (+3/-3)
- `example/ck_tile/99_toy_example/04_codegen_flash_attention_fwd/block_gemm_pipeline_agmem_bgmem_creg_v2_askiplds.hpp`  (+3/-3)
- `include/ck_tile/core/tensor/slice_tile.hpp`  (+4/-2)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1.hpp`  (+2/-1)
- `example/ck_tile/99_toy_example/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/99_toy_example/00_add_vector/add_vector.cpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("m", "256000000", "m dimension")
.insert("v", "1", "cpu validation or not")
```

**`example/ck_tile/99_toy_example/00_add_vector/add_vector.hpp`**
```
namespace ck_tile {
template <typename BlockWarps, typename BlockTile, typename WarpTile, typename Vector>
struct AddVectorShape
static constexpr index_t Block_M = BlockTile::at(number<0>{});
```

**`example/ck_tile/99_toy_example/00_add_vector/reference_add_vector.hpp`**
```
namespace ck_tile {
template <typename XDataType, typename YDataType>
CK_TILE_HOST void reference_add_vector(const HostTensor<XDataType>& xa_m_n,
const HostTensor<XDataType>& xb_m_n,
```

**`example/ck_tile/99_toy_example/01_add/add.cpp`**
```
using BlockWarps =
ck_tile::sequence<1, 8>; // number of concurrent warps in one block (if 8 warps * 64 threads
using BlockTile =
ck_tile::sequence<1, 4096>; // shape of one blockTile (elements covered by one block)
```

**`example/ck_tile/99_toy_example/01_add/add.hpp`**
```
static constexpr index_t Block_M = BlockTile::at(number<0>{}); // elements along M in one Block
static constexpr index_t Block_N = BlockTile::at(number<1>{}); // elements along N in one Block
static constexpr index_t Warp_M = WarpTile::at(number<0>{}); // elements along M in one Warp
static constexpr index_t Warp_N = WarpTile::at(number<1>{}); // elements along N in one Warp
```
