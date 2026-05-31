# Diff summary

- **files changed:** 32
- **lines:** +883 / -306
- **kernel-ish files:** 32

## Files (by churn)

- `include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`  (+151/-5)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+80/-41)
- `include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`  (+97/-21)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+84/-12)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+60/-8)
- `include/ck_tile/core/numeric/vector_type.hpp`  (+43/-23)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+37/-28)
- `include/ck_tile/core/tensor/tile_window_linear.hpp`  (+34/-23)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+28/-19)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+29/-16)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+25/-15)
- `include/ck_tile/host/check_err.hpp`  (+16/-14)
- `include/ck_tile/host/host_tensor.hpp`  (+15/-13)
- `include/ck_tile/core/numeric/half.hpp`  (+21/-3)
- `include/ck_tile/core/tensor/static_distributed_tensor.hpp`  (+13/-9)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.hpp`**
```
template <typename ADataType, typename BDataType = ADataType, typename CDataType = ADataType>
template <>
struct GemmBasicTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>
using ADataType   = ck_tile::half_t;
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
template <typename Tensor>
void permute_tensor_b(Tensor& tensor)
const ck_tile::index_t K = tensor.get_length(0);
const ck_tile::index_t N = tensor.get_length(1);
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
else if(data_type == "pk_int4_t")
return run_gemm_example_with_layouts<ck_tile::half_t,
ck_tile::pk_int4_t,
ck_tile::half_t>(argc, argv, Row{}, Col{}, Row{});
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
(std::is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
(std::is_same<T, pk_int4_t>::value &&
(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)),
```

**`include/ck_tile/core/container/array.hpp`**
```
template <typename, typename>
struct vector_traits<array<T, N>, void>
```
