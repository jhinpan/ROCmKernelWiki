# Diff summary

- **files changed:** 14
- **lines:** +100 / -450
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck_tile/ops/common/generic_2d_block_shape.hpp`  (+50/-24)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`  (+5/-43)
- `test/ck_tile/add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`  (+5/-43)
- `example/ck_tile/12_smoothquant/smoothquant.hpp`  (+4/-42)
- `test/ck_tile/moe_smoothquant/moe_smoothquant.hpp`  (+4/-42)
- `example/ck_tile/14_moe_smoothquant/moe_smoothquant.hpp`  (+4/-41)
- `test/ck_tile/smoothquant/smoothquant.hpp`  (+4/-41)
- `example/ck_tile/10_rmsnorm2d/generate.py`  (+3/-40)
- `test/ck_tile/layernorm2d/generate.py`  (+3/-40)
- `test/ck_tile/rmsnorm2d/generate.py`  (+3/-40)
- `example/ck_tile/02_layernorm2d/generate.py`  (+2/-39)
- `example/ck_tile/10_rmsnorm2d/example_rmsnorm2d_fwd.cpp`  (+5/-5)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/example_add_rmsnorm2d_rdquant_fwd.cpp`  (+4/-5)
- `example/ck_tile/12_smoothquant/example_smoothquant.cpp`  (+4/-5)

## Key added lines (kernel files)

**`example/ck_tile/02_layernorm2d/generate.py`**
```
using ThreadPerBlock = ck_tile::sequence<ThreadPerBlock_M_, ThreadPerBlock_N_>;
using Shape = ck_tile::Generic2dBlockShape<BlockTile, ThreadPerBlock, Vector>;
```

**`example/ck_tile/10_rmsnorm2d/example_rmsnorm2d_fwd.cpp`**
```
using BlockTile      = ck_tile::sequence<2, 128>;
using Vector         = ck_tile::sequence<1, 1>;
using ThreadPerBlock = ck_tile::sequence<2, 128>;
using Shape = ck_tile::Generic2dBlockShape<BlockTile, ThreadPerBlock, Vector>;
```

**`example/ck_tile/10_rmsnorm2d/generate.py`**
```
using ThreadPerBlock = ck_tile::sequence<ThreadPerBlock_M_, ThreadPerBlock_N_>;
using Shape = ck_tile::Generic2dBlockShape<BlockTile, ThreadPerBlock, Vector>;
```

**`example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`**
```
using BlockTile      = ck_tile::sequence<Block_M, Block_N>;
using Vector         = ck_tile::sequence<1, Vector_N_>;
using ThreadPerBlock = ck_tile::sequence<ThreadPerBlock_M_, ThreadPerBlock_N_>;
using Shape = ck_tile::Generic2dBlockShape<BlockTile, ThreadPerBlock, Vector>;
```

**`example/ck_tile/11_add_rmsnorm2d_rdquant/example_add_rmsnorm2d_rdquant_fwd.cpp`**
```
using BlockTile      = ck_tile::sequence<4, 128>;
using Vector         = ck_tile::sequence<1, 1>;
using ThreadPerBlock = ck_tile::sequence<4, 64>;
using Shape   = ck_tile::Generic2dBlockShape<BlockTile, ThreadPerBlock, Vector>;
```
