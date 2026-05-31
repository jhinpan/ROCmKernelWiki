# Diff summary

- **files changed:** 10
- **lines:** +343 / -1288
- **kernel-ish files:** 8

## Files (by churn)

- `example/ck_tile/38_block_scale_gemm/run_gemm_aquant_example.inc`  (+0/-302)
- `example/ck_tile/38_block_scale_gemm/run_gemm_bquant_example.inc`  (+0/-296)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+170/-79)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_preshuffle.cpp`  (+0/-234)
- `example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`  (+116/-116)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_basic.cpp`  (+0/-231)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+32/-14)
- `include/ck_tile/ops/gemm_group_quant/kernel/gemm_quant_kernel.hpp`  (+14/-10)
- `example/ck_tile/38_block_scale_gemm/README.md`  (+11/-0)
- `example/ck_tile/38_block_scale_gemm/CMakeLists.txt`  (+0/-6)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`**
```
ck_tile::QuantType QuantMode,
typename CDEElementWise>
using ComputeDataType = std::conditional_t<QuantMode == ck_tile::QuantType::AQuantGrouped ||
QuantMode == ck_tile::QuantType::RowColQuant,
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
ck_tile::tensor_layout::gemm::RowMajor>>{};
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
auto shuffle_aq(const ck_tile::HostTensor<T>* t, int block_aq_k)
if(t->get_lengths().size() != 2)
int m_   = t->get_lengths()[0];
int aqk_ = t->get_lengths()[1];
```

**`include/ck_tile/ops/gemm_group_quant/kernel/gemm_quant_kernel.hpp`**
```
template <typename T>
concept HasStaticPreshuffleQuant = requires {
{ T::PreshuffleQuant } -> std::convertible_to<decltype(T::PreshuffleQuant)>;
template <typename T>
```
