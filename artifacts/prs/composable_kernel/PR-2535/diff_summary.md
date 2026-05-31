# Diff summary

- **files changed:** 14
- **lines:** +905 / -199
- **kernel-ish files:** 12

## Files (by churn)

- `test/ck_tile/reduce/test_reduce2d.cpp`  (+359/-0)
- `include/ck_tile/ops/reduce/kernel/reduce2d_kernel.hpp`  (+219/-0)
- `example/ck_tile/05_reduce/reduce.hpp`  (+0/-164)
- `include/ck_tile/host/reference/reference_reduce.hpp`  (+78/-0)
- `include/ck_tile/ops/reduce/block/block_reduce2d.hpp`  (+63/-9)
- `example/ck_tile/05_reduce/reduce.cpp`  (+48/-15)
- `include/ck_tile/core/utility/reduce_operator.hpp`  (+52/-5)
- `include/ck_tile/ops/reduce/pipeline/reduce2d_shape.hpp`  (+37/-0)
- `include/ck_tile/ops/reduce/pipeline/reduce2d_problem.hpp`  (+27/-0)
- `include/ck_tile/ops/reduce/pipeline/reduce2d_default_policy.hpp`  (+5/-4)
- `test/ck_tile/reduce/CMakeLists.txt`  (+7/-0)
- `include/ck_tile/core/container/thread_buffer.hpp`  (+5/-1)
- `include/ck_tile/ops/reduce.hpp`  (+4/-1)
- `test/ck_tile/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/05_reduce/reduce.cpp`**
```
arg_parser.insert("n", "32", "n dimension")
.insert("h", "7", "h dimension")
.insert("w", "7", "w dimension")
.insert("c", "512", "c dimension")
```

**`include/ck_tile/core/container/thread_buffer.hpp`**
```
CK_TILE_HOST_DEVICE constexpr thread_buffer(const value_type & o) : data{} {
static_for<0, N, 1>{}(
[&](auto i) { data[i] = o; }
```

**`include/ck_tile/core/utility/reduce_operator.hpp`**
```
typename = std::enable_if_t<std::is_same_v<T, half_t> || std::is_same_v<T, bf16_t> ||
std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>>>
static constexpr bool requires_special_combine = false;
template <typename T,
```

**`include/ck_tile/host/reference/reference_reduce.hpp`**
```
template <
typename XDataType,
typename ComputeDataType,
typename YDataType,
```

**`include/ck_tile/ops/reduce/block/block_reduce2d.hpp`**
```
template <
typename XDistributedTensor_,
typename YDistributedTensor_,
typename ReduceFunc,
```
