# Diff summary

- **files changed:** 19
- **lines:** +1423 / -47
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck_tile/host/convolution_parameter.hpp`  (+283/-0)
- `include/ck_tile/host/convolution_host_tensor_descriptor_helper.hpp`  (+266/-0)
- `include/ck_tile/ops/image_to_column/kernel/image_to_column_kernel.hpp`  (+224/-0)
- `example/ck_tile/04_img2col/image_to_column.cpp`  (+170/-0)
- `include/ck_tile/host/reference/reference_im2col.hpp`  (+117/-45)
- `test/ck_tile/image_to_column/test_tile_image_to_column.cpp`  (+142/-0)
- `example/ck_tile/04_img2col/image_to_column.hpp`  (+105/-0)
- `include/ck_tile/ops/image_to_column/pipeline/tile_image_to_column_shape.hpp`  (+32/-0)
- `include/ck_tile/ops/image_to_column/pipeline/block_image_to_column_problem.hpp`  (+27/-0)
- `include/ck_tile/host/host_tensor.hpp`  (+14/-1)
- `example/ck_tile/04_img2col/README.md`  (+12/-0)
- `include/ck_tile/core/container/array.hpp`  (+11/-1)
- `include/ck_tile/ops/image_to_column.hpp`  (+8/-0)
- `test/ck_tile/image_to_column/CMakeLists.txt`  (+4/-0)
- `example/ck_tile/04_img2col/CMakeLists.txt`  (+3/-0)

## Key added lines (kernel files)

**`example/ck_tile/04_img2col/image_to_column.cpp`**
```
template <>
float image_to_column(const image_to_column_traits& traits,
const image_to_column_args<2>& args,
const ck_tile::stream_config& stream_conf)
```

**`example/ck_tile/04_img2col/image_to_column.hpp`**
```
ck_tile::conv::ConvParam                                                 \
{                                                                        \
2, 2, 32, 32, 32, {4, 4}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, { 0, 0 } \
struct ExecutionConfig final
```

**`include/ck_tile/core/container/array.hpp`**
```
template <typename T, index_t N, typename X>
CK_TILE_HOST_DEVICE constexpr auto to_array(const std::vector<X>& x)
array<T, N> arr;
static_for<0, N, 1>{}([&x, &arr](auto i) { arr(i) = x[i]; });
```

**`include/ck_tile/host/convolution_host_tensor_descriptor_helper.hpp`**
```
namespace ck_tile {
namespace conv {
namespace detail {
template <typename OldLayout>
```

**`include/ck_tile/host/convolution_parameter.hpp`**
```
namespace ck_tile {
namespace conv {
struct ConvParam
ConvParam();
```
