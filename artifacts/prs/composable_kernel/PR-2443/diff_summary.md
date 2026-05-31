# Diff summary

- **files changed:** 28
- **lines:** +1205 / -525
- **kernel-ish files:** 24

## Files (by churn)

- `include/ck_tile/core/algorithm/coordinate_transform.hpp`  (+179/-236)
- `include/ck_tile/core/tensor/tile_distribution_encoding.hpp`  (+102/-102)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+21/-88)
- `test/ck_tile/utility/print/test_print_static_encoding_pattern.cpp`  (+89/-0)
- `test/ck_tile/utility/print/test_print_coordinate_transform.cpp`  (+83/-0)
- `test/ck_tile/utility/print/test_print_buffer_view.cpp`  (+78/-0)
- `include/ck_tile/core/utility/print.hpp`  (+76/-0)
- `test/ck_tile/utility/print/test_print_basic_types.cpp`  (+76/-0)
- `test/ck_tile/utility/print/README.md`  (+70/-0)
- `test/ck_tile/utility/print/test_print_tuple.cpp`  (+66/-0)
- `include/ck_tile/core/tensor/tensor_adaptor.hpp`  (+34/-31)
- `test/ck_tile/utility/print/test_print_array.cpp`  (+59/-0)
- `include/ck_tile/core/algorithm/static_encoding_pattern.hpp`  (+48/-0)
- `test/ck_tile/utility/print/test_print_sequence.cpp`  (+45/-0)
- `include/ck_tile/core/tensor/tensor_descriptor.hpp`  (+27/-15)

## Key added lines (kernel files)

**`include/ck_tile/core/algorithm/coordinate_transform.hpp`**
```
template <typename LowLength>
CK_TILE_HOST_DEVICE static void print(const pass_through<LowLength>& pt)
printf("pass_through{");
printf("up_lengths_: ");
```

**`include/ck_tile/core/algorithm/static_encoding_pattern.hpp`**
```
constexpr const char* tile_distribution_pattern_to_string(tile_distribution_pattern pattern)
switch(pattern)
case tile_distribution_pattern::thread_raked: return "thread_raked";
case tile_distribution_pattern::warp_raked: return "warp_raked";
```

**`include/ck_tile/core/arch/arch.hpp`**
```
CK_TILE_HOST_DEVICE constexpr const char* address_space_to_string(address_space_enum addr_space)
switch(addr_space)
case address_space_enum::generic: return "generic";
case address_space_enum::global: return "global";
```

**`include/ck_tile/core/container/array.hpp`**
```
template <typename T, index_t N>
CK_TILE_HOST_DEVICE static void print(const array<T, N>& a)
printf("array{size: %ld, data: [", static_cast<long>(N));
for(index_t i = 0; i < N; ++i)
```

**`include/ck_tile/core/container/map.hpp`**
```
template <typename key, typename data, index_t max_size>
CK_TILE_HOST_DEVICE static void print(const map<key, data, max_size>& m)
printf("map{size_: %d, impl_: [", m.size_);
for(const auto& [k, d] : m)
```
