# Diff summary

- **files changed:** 12
- **lines:** +1225 / -13
- **kernel-ish files:** 10

## Files (by churn)

- `example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`  (+443/-0)
- `include/ck_tile/ops/gemm_group_quant/kernel/grouped_gemm_quant_kernel.hpp`  (+433/-0)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`  (+157/-0)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`  (+136/-0)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+30/-0)
- `include/ck_tile/core/tensor/tensor_descriptor.hpp`  (+9/-3)
- `include/ck_tile/ops/gemm_group_quant/kernel/gemm_quant_kernel.hpp`  (+5/-6)
- `include/ck_tile/ops/gemm_group_quant/pipeline/tile_gemm_quant_traits.hpp`  (+7/-3)
- `example/ck_tile/17_grouped_gemm/README.md`  (+2/-0)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+1/-1)
- `example/ck_tile/17_grouped_gemm/CMakeLists.txt`  (+1/-0)
- `include/ck_tile/ops/gemm_group_quant.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
template <template <typename PrecType> typename GemmConfig>
int run_grouped_gemm_example(int argc, char* argv[])
auto [result, arg_parser] = create_args(argc, argv);
if(!result)
```

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`**
```
template <typename GemmConfig,
typename ALayout,
typename AQLayout,
typename BLayout,
```

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`**
```
template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile()
constexpr bool is_8bit_float =
std::is_same_v<PrecType, ck_tile::fp8_t> || std::is_same_v<PrecType, ck_tile::bf8_t>;
```

**`example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`**
```
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
ck_tile::tensor_layout::gemm::RowMajor>>{};
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`**
```
stride_As[i] = ck_tile::get_default_stride(M, K, stride_As[i], is_row_major(a_layout));
```
