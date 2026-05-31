# Diff summary

- **files changed:** 14
- **lines:** +426 / -75
- **kernel-ish files:** 13

## Files (by churn)

- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_util_quant.hpp`  (+70/-27)
- `include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`  (+79/-4)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`  (+38/-7)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant.cpp`  (+22/-20)
- `example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`  (+33/-5)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_rowcol.cpp`  (+35/-0)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_tensor.cpp`  (+35/-0)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_bquant.cpp`  (+33/-0)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_ut_cases.inc`  (+28/-2)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_v2.hpp`  (+27/-0)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`  (+12/-7)
- `test/ck_tile/grouped_gemm_quant/CMakeLists.txt`  (+9/-2)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+4/-0)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`**
```
GemmConfig::PreshuffleB, // PreshuffleB
true>; // Persistence
using GemmPipeline = std::conditional_t<
QuantMode == ck_tile::QuantType::RowColQuant ||
```

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`**
```
template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_from_preshuffled_warp_tile()
if constexpr(M_Warp_Tile == 32)
return sizeof(PrecType) == 2 ? 16 : 64;
```

**`example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`**
```
const int init_method                 = arg_parser.get_int("init");
else if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
stride_AQs[i] = 0; // No A quantization
stride_BQs[i] =
```

**`include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp`**
```
CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
```

**`include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`**
```
if constexpr(GemmPipeline::DoubleSmemBuffer == true &&
kQuantType == QuantType::BQuantGrouped)
__shared__ char smem_ptr_1[GetSmemSize()];
RunGemmWithPipelineSelection2LDS(a_ptr,
```
