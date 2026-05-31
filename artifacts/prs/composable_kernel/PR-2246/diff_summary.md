# Diff summary

- **files changed:** 23
- **lines:** +1509 / -6
- **kernel-ish files:** 17

## Files (by churn)

- `test/ck_tile/container/test_tuple_apply.cpp`  (+223/-0)
- `test/ck_tile/elementwise/test_elementwise_1d.cpp`  (+216/-0)
- `example/ck_tile/21_elementwise/elementwise_example.cpp`  (+214/-0)
- `example/ck_tile/21_elementwise/elementwise_example_add_4d.cpp`  (+159/-0)
- `example/ck_tile/21_elementwise/elementwise_example_transpose.cpp`  (+156/-0)
- `example/ck_tile/21_elementwise/elementwise_example_unary.cpp`  (+147/-0)
- `include/ck_tile/ops/elementwise/kernel/elementwise_kernel.hpp`  (+123/-0)
- `include/ck_tile/ops/elementwise/binary_elementwise_operation.hpp`  (+94/-0)
- `include/ck_tile/host/reference/reference_transpose.hpp`  (+33/-0)
- `include/ck_tile/ops/elementwise/pipeline/elementwise_pipeline_default_policy.hpp`  (+29/-0)
- `include/ck_tile/ops/elementwise/pipeline/elementwise_shape.hpp`  (+29/-0)
- `include/ck_tile/ops/elementwise/pipeline/elementwise_pipeline_problem.hpp`  (+26/-0)
- `include/ck_tile/core/container/tuple.hpp`  (+21/-4)
- `example/ck_tile/21_elementwise/CMakeLists.txt`  (+15/-0)
- `test/ck_tile/container/CMakeLists.txt`  (+6/-0)

## Key added lines (kernel files)

**`example/ck_tile/21_elementwise/elementwise_example.cpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("m", "1024", "m dimension")
.insert("n", "1024", "n dimension")
```

**`example/ck_tile/21_elementwise/elementwise_example_add_4d.cpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("dim0", "4", "dimension 0")
.insert("dim1", "16", "dimension 1")
```

**`example/ck_tile/21_elementwise/elementwise_example_transpose.cpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("m", "1024", "m dimension of input")
.insert("n", "1024", "n dimension of input")
```

**`example/ck_tile/21_elementwise/elementwise_example_unary.cpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("m", "1024", "m dimension")
.insert("n", "1024", "n dimension")
```

**`include/ck_tile/core/container/tuple.hpp`**
```
template<index_t I> CK_TILE_HOST_DEVICE constexpr decltype(auto) get() const &          { TP_COM_(); return impl::getv<I
template<index_t I> CK_TILE_HOST_DEVICE constexpr decltype(auto) get(number<I>) const & { TP_COM_(); return get<I>(); }
template<index_t I> CK_TILE_HOST_DEVICE constexpr decltype(auto) get() &                { TP_COM_(); return impl::getv<I
template<index_t I> CK_TILE_HOST_DEVICE constexpr decltype(auto) get(number<I>) &       { TP_COM_(); return get<I>(); }
```
