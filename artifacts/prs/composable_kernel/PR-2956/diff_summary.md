# Diff summary

- **files changed:** 14
- **lines:** +1317 / -0
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck_tile/ops/pooling/kernel/pool_kernel.hpp`  (+496/-0)
- `test/ck_tile/pooling/test_pooling.cpp`  (+249/-0)
- `example/ck_tile/36_pooling/pool3d.cpp`  (+188/-0)
- `include/ck_tile/host/reference/reference_pool.hpp`  (+147/-0)
- `include/ck_tile/ops/pooling/pipeline/pool_default_policy.hpp`  (+80/-0)
- `include/ck_tile/ops/pooling/pipeline/pool_shape.hpp`  (+57/-0)
- `example/ck_tile/36_pooling/README.md`  (+42/-0)
- `include/ck_tile/ops/pooling/pipeline/pool_problem.hpp`  (+33/-0)
- `include/ck_tile/ops/pool.hpp`  (+11/-0)
- `example/ck_tile/36_pooling/CMakeLists.txt`  (+8/-0)
- `test/ck_tile/pooling/CMakeLists.txt`  (+3/-0)
- `CHANGELOG.md`  (+1/-0)
- `example/ck_tile/CMakeLists.txt`  (+1/-0)
- `test/ck_tile/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/36_pooling/pool3d.cpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("N", "2", "N dimension")
.insert("H", "30", "H dimension")
```

**`include/ck_tile/host/reference/reference_pool.hpp`**
```
namespace ck_tile {
template <typename InDataType,
typename ComputeDataType,
typename OutDataType,
```

**`include/ck_tile/ops/pooling/kernel/pool_kernel.hpp`**
```
namespace ck_tile {
template <typename TensorShape, typename WindowShape>
struct PoolHostArgs
CK_TILE_HOST PoolHostArgs(const void* input_ptr_,
```

**`include/ck_tile/ops/pooling/pipeline/pool_default_policy.hpp`**
```
namespace ck_tile {
struct PoolDefaultPolicy
template <typename Problem>
CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
```

**`include/ck_tile/ops/pooling/pipeline/pool_problem.hpp`**
```
namespace ck_tile {
template <typename InDataType_,
typename OutDataType_,
typename ComputeDataType_,
```
