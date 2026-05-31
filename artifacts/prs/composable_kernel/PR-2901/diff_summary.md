# Diff summary

- **files changed:** 11
- **lines:** +1802 / -0
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck_tile/ops/batched_contraction/kernel/batched_contraction_kernel.hpp`  (+522/-0)
- `example/ck_tile/41_batched_contraction/run_batched_contraction_example.inc`  (+405/-0)
- `include/ck_tile/host/reference/reference_batched_contraction.hpp`  (+265/-0)
- `example/ck_tile/41_batched_contraction/batched_contraction.cpp`  (+245/-0)
- `include/ck_tile/ops/batched_contraction/utils/tensor_descriptor_utils.hpp`  (+169/-0)
- `example/ck_tile/41_batched_contraction/contraction_utils.hpp`  (+146/-0)
- `include/ck_tile/ops/batched_contraction/pipeline/batched_contraction_problem.hpp`  (+32/-0)
- `include/ck_tile/ops/batched_contraction.hpp`  (+9/-0)
- `example/ck_tile/41_batched_contraction/CMakeLists.txt`  (+7/-0)
- `CHANGELOG.md`  (+1/-0)
- `example/ck_tile/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/41_batched_contraction/batched_contraction.cpp`**
```
template <typename ADataType,
typename BDataType,
typename DsDataType,
typename AccDataType,
```

**`example/ck_tile/41_batched_contraction/contraction_utils.hpp`**
```
struct AddDs
template <typename E, typename C, typename... Ds>
CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const Ds&... ds) const -> void
const float x0_f =
```

**`example/ck_tile/41_batched_contraction/run_batched_contraction_example.inc`**
```
template <typename ADataType, typename BDataType, typename EDataType, typename AccDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
const ck_tile::index_t kbatch,
const float max_accumulated_value)
```

**`include/ck_tile/host/reference/reference_batched_contraction.hpp`**
```
namespace ck_tile {
template <typename ADataType,
typename BDataType,
typename DDataType,
```

**`include/ck_tile/ops/batched_contraction/kernel/batched_contraction_kernel.hpp`**
```
namespace ck_tile {
template <ck_tile::index_t NumDTensor = 0>
struct BatchedContractionHostArgs
CK_TILE_HOST
```
