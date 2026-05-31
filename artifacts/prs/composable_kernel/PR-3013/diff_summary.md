# Diff summary

- **files changed:** 13
- **lines:** +859 / -98
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck_tile/ops/reduce/block/block_reduce2d.hpp`  (+311/-25)
- `test/ck_tile/pooling/test_pooling.cpp`  (+200/-11)
- `include/ck_tile/ops/pooling/kernel/pool_kernel.hpp`  (+89/-10)
- `include/ck_tile/core/utility/reduce_operator.hpp`  (+51/-28)
- `include/ck_tile/host/reference/reference_pool.hpp`  (+55/-5)
- `include/ck_tile/core/utility/reduce_operator_accumulate.hpp`  (+50/-0)
- `example/ck_tile/36_pooling/pool3d.cpp`  (+37/-9)
- `include/ck_tile/ops/pooling/pipeline/pool_default_policy.hpp`  (+28/-4)
- `include/ck_tile/ops/reduce/pipeline/reduce2d_default_policy.hpp`  (+26/-4)
- `include/ck_tile/ops/reduce/block/block_reduce2d_problem.hpp`  (+6/-1)
- `include/ck_tile/ops/reduce/pipeline/reduce2d_problem.hpp`  (+3/-1)
- `include/ck_tile/ops/pooling/pipeline/pool_problem.hpp`  (+2/-0)
- `include/ck_tile/core.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/36_pooling/pool3d.cpp`**
```
template <typename InDataType,
typename OutDataType,
typename ComputeDataType,
typename IndexDataType>
```

**`include/ck_tile/core/utility/reduce_operator.hpp`**
```
typename = std::enable_if_t<is_any_of<T, float, double, int32_t, int8_t>::value>>
typename = std::enable_if_t<is_any_of<T, half_t, bf16_t, fp8_t, bf8_t>::value>>
typename = std::enable_if_t<is_any_of<T, float, double, int32_t, int8_t>::value>>
typename = std::enable_if_t<is_any_of<T, half_t, bf16_t, fp8_t, bf8_t>::value>>
```

**`include/ck_tile/core/utility/reduce_operator_accumulate.hpp`**
```
namespace ck_tile {
struct AccumulateWithIndex
template <typename ReduceOp, typename T, typename IndexType>
CK_TILE_HOST_DEVICE void operator()(const ReduceOp& reduce_func,
```

**`include/ck_tile/host/reference/reference_pool.hpp`**
```
typename IndexDataType,
typename WindowShape,
bool OutputIndex = false>
HostTensor<IndexDataType>& output_index,
```

**`include/ck_tile/ops/pooling/kernel/pool_kernel.hpp`**
```
void* output_index_ptr_,
output_index_ptr(output_index_ptr_),
void* output_index_ptr;
void* output_index_ptr;
```
