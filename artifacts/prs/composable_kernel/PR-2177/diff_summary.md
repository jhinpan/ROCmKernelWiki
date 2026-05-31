# Diff summary

- **files changed:** 17
- **lines:** +1523 / -1
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck_tile/core/tensor/load_tile_transpose.hpp`  (+362/-0)
- `example/ck_tile/37_transpose/transpose_example.cpp`  (+257/-0)
- `example/ck_tile/37_transpose/transpose_policy.hpp`  (+151/-0)
- `example/ck_tile/37_transpose/block_transpose.hpp`  (+149/-0)
- `example/ck_tile/37_transpose/batched_transpose_kernel.hpp`  (+120/-0)
- `include/ck_tile/core/arch/amd_transpose_load_encoding.hpp`  (+86/-0)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+82/-0)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+76/-1)
- `example/ck_tile/37_transpose/transpose_api.cpp`  (+59/-0)
- `include/ck_tile/core/tensor/tile_window_linear.hpp`  (+54/-0)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+34/-0)
- `example/ck_tile/37_transpose/README.md`  (+27/-0)
- `example/ck_tile/37_transpose/transpose_example.hpp`  (+27/-0)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+27/-0)
- `example/ck_tile/37_transpose/CMakeLists.txt`  (+9/-0)

## Key added lines (kernel files)

**`example/ck_tile/37_transpose/batched_transpose_kernel.hpp`**
```
namespace ck_tile {
struct BatchedTransposeHostArgs
const void* p_input;
void* p_output;
```

**`example/ck_tile/37_transpose/block_transpose.hpp`**
```
namespace ck_tile {
template <typename Layout_, index_t kRow, index_t kCol>
struct TransposeTraits
static constexpr index_t kLeadDim   = kCol;
```

**`example/ck_tile/37_transpose/transpose_api.cpp`**
```
template <typename ts_type,
ck_tile::index_t block_x,
ck_tile::index_t block_y,
ck_tile::index_t warp_x,
```

**`example/ck_tile/37_transpose/transpose_example.cpp`**
```
template <typename T>
void dump_host_tensor_4d(const ck_tile::HostTensor<T>& x)
auto len = x.get_lengths();
assert(len.size() == 4);
```

**`example/ck_tile/37_transpose/transpose_example.hpp`**
```
struct batched_transpose_trait
std::string type;
std::string layout;
struct batched_transpose_kargs : public ck_tile::BatchedTransposeHostArgs
```
