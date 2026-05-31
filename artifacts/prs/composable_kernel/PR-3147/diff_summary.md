# Diff summary

- **files changed:** 26
- **lines:** +2661 / -2
- **kernel-ish files:** 20

## Files (by churn)

- `include/ck_tile/ops/reduce/kernel/multi_reduce2d_kernel.hpp`  (+363/-0)
- `test/ck_tile/reduce/test_multi_reduce2d_multiblock_impl.hpp`  (+355/-0)
- `test/ck_tile/reduce/test_multi_reduce2d_threadwise_impl.hpp`  (+325/-0)
- `example/ck_tile/05_reduce/multiple_reduce_multiblock.cpp`  (+271/-0)
- `include/ck_tile/host/reference/reference_reduce.hpp`  (+230/-0)
- `example/ck_tile/05_reduce/multiple_reduce_threadwise.cpp`  (+224/-0)
- `tile_engine/ops/reduce/reduce_instance_builder.py`  (+171/-0)
- `tile_engine/ops/reduce/reduce_parameter.py`  (+127/-0)
- `tile_engine/ops/reduce/CMakeLists.txt`  (+126/-0)
- `include/ck_tile/ops/reduce/kernel/multi_reduce2d_tile_partitioner.hpp`  (+125/-0)
- `test/ck_tile/reduce/test_multi_reduce2d_threadwise.cpp`  (+96/-0)
- `test/ck_tile/reduce/test_multi_reduce2d_multiblock.cpp`  (+91/-0)
- `tile_engine/ops/reduce/configs/default_multi_reduce_config.json`  (+51/-0)
- `test/ck_tile/reduce/test_multi_reduce2d_common.hpp`  (+34/-0)
- `example/ck_tile/05_reduce/CMakeLists.txt`  (+16/-0)

## Key added lines (kernel files)

**`example/ck_tile/05_reduce/multiple_reduce_multiblock.cpp`**
```
template <typename T>
struct DataTypeTraits;
template <>
struct DataTypeTraits<ck_tile::half_t>
```

**`example/ck_tile/05_reduce/multiple_reduce_threadwise.cpp`**
```
template <typename T>
struct DataTypeTraits;
template <>
struct DataTypeTraits<ck_tile::half_t>
```

**`include/ck_tile/core/utility/reduce_operator.hpp`**
```
CK_TILE_HOST_DEVICE static constexpr auto GetAtomic()
return memory_operation_enum::atomic_add;
```

**`include/ck_tile/host/reference/reference_reduce.hpp`**
```
template <typename XDataType,
typename ComputeDataType,
typename YDataType,
typename YRefTuple,
```

**`include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`**
```
template <typename Y, typename X>
CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const
y = ck_tile::type_convert<raw_t<Y>>(x);
```
