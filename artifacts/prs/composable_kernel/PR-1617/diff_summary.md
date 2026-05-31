# Diff summary

- **files changed:** 62
- **lines:** +1758 / -219
- **kernel-ish files:** 51

## Files (by churn)

- `example/ck_tile/12_smoothquant/example_smoothquant.cpp`  (+237/-0)
- `example/ck_tile/12_smoothquant/smoothquant.cpp`  (+218/-0)
- `include/ck_tile/ops/smoothquant/kernel/smoothquant_kernel.hpp`  (+176/-0)
- `example/ck_tile/12_smoothquant/instances/smoothquant_fwd_api.cpp`  (+143/-0)
- `include/ck_tile/ops/smoothquant/pipeline/smoothquant_pipeline_two_pass.hpp`  (+132/-0)
- `example/ck_tile/12_smoothquant/smoothquant.hpp`  (+114/-0)
- `include/ck_tile/ops/smoothquant/pipeline/smoothquant_pipeline_default_policy.hpp`  (+95/-0)
- `include/ck_tile/ops/smoothquant/pipeline/smoothquant_pipeline_one_pass.hpp`  (+94/-0)
- `include/ck_tile/ops/add_rmsnorm2d_rdquant/kernel/add_rmsnorm2d_rdquant_fwd_shape.hpp`  (+0/-78)
- `include/ck_tile/ops/rmsnorm2d/kernel/rmsnorm2d_fwd_shape.hpp`  (+0/-78)
- `example/ck_tile/12_smoothquant/instances/smoothquant_instance_common.hpp`  (+62/-0)
- `example/ck_tile/12_smoothquant/script/perf_test.sh`  (+37/-0)
- `include/ck_tile/ops/smoothquant/pipeline/smoothquant_pipeline_problem.hpp`  (+35/-0)
- `example/ck_tile/12_smoothquant/script/smoke_test.sh`  (+30/-0)
- `example/ck_tile/12_smoothquant/CMakeLists.txt`  (+24/-0)

## Key added lines (kernel files)

**`example/ck_tile/10_rmsnorm2d/example_rmsnorm2d_fwd.cpp`**
```
using Shape   = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;
```

**`example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.hpp`**
```
using Shape = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;
```

**`example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`**
```
using YScaleDataType  = float;
using YScaleDataType  = float;
using Shape = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;
```

**`example/ck_tile/11_add_rmsnorm2d_rdquant/example_add_rmsnorm2d_rdquant_fwd.cpp`**
```
using YScaleDataType  = float;
using BlockWarps = ck_tile::sequence<4, 1>;
using BlockTile  = ck_tile::sequence<4, 128>;
using Shape   = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;
```

**`example/ck_tile/12_smoothquant/example_smoothquant.cpp`**
```
template <typename DataType>
auto get_elimit()
double rtol = 1e-5;
double atol = 1e-5;
```
