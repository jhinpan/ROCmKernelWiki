# Diff summary

- **files changed:** 14
- **lines:** +760 / -0
- **kernel-ish files:** 10

## Files (by churn)

- `example/ck_tile/35_batched_transpose/batched_transpose_example.cpp`  (+261/-0)
- `include/ck_tile/ops/batched_transpose/kernel/batched_transpose_kernel.hpp`  (+129/-0)
- `example/ck_tile/35_batched_transpose/batched_transpose_api.cpp`  (+82/-0)
- `include/ck_tile/host/reference/reference_batched_transpose.hpp`  (+59/-0)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_pipeline.hpp`  (+52/-0)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_problem.hpp`  (+48/-0)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_policy.hpp`  (+44/-0)
- `example/ck_tile/35_batched_transpose/README.md`  (+27/-0)
- `example/ck_tile/35_batched_transpose/batched_transpose_example.hpp`  (+25/-0)
- `example/ck_tile/35_batched_transpose/script/smoke_test.sh`  (+11/-0)
- `include/ck_tile/ops/batched_transpose.hpp`  (+11/-0)
- `example/ck_tile/35_batched_transpose/CMakeLists.txt`  (+9/-0)
- `example/ck_tile/CMakeLists.txt`  (+1/-0)
- `include/ck_tile/host.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/35_batched_transpose/batched_transpose_api.cpp`**
```
template <typename ts_type,
ck_tile::index_t block_x,
ck_tile::index_t block_y,
ck_tile::index_t warp_x,
```

**`example/ck_tile/35_batched_transpose/batched_transpose_example.cpp`**
```
template <typename T>
void dump_host_tensor_4d(const ck_tile::HostTensor<T>& x)
auto len = x.get_lengths();
assert(len.size() == 4);
```

**`example/ck_tile/35_batched_transpose/batched_transpose_example.hpp`**
```
struct batched_transpose_trait
std::string type;
std::string layout;
struct batched_transpose_kargs : public ck_tile::BatchedTransposeHostArgs
```

**`include/ck_tile/host/reference/reference_batched_transpose.hpp`**
```
namespace ck_tile {
template <typename Type>
CK_TILE_HOST void reference_batched_transpose(const HostTensor<Type>& x,
HostTensor<Type>& y,
```

**`include/ck_tile/ops/batched_transpose/kernel/batched_transpose_kernel.hpp`**
```
namespace ck_tile {
struct BatchedTransposeHostArgs
const void* p_input;
void* p_output;
```
