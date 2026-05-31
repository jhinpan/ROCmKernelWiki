# Diff summary

- **files changed:** 7
- **lines:** +257 / -36
- **kernel-ish files:** 7

## Files (by churn)

- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb.cpp`  (+175/-17)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffle.cpp`  (+43/-1)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+18/-9)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_flatbr_bquant_cr.hpp`  (+11/-2)
- `include/ck_tile/host/tensor_shuffle_utils.hpp`  (+7/-3)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+3/-3)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+0/-1)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb.cpp`**
```
using TypeConfig     = decltype(GemmQuantTypeConfig<ck_tile::fp8_t,
ck_tile::fp8_t,
ck_tile::half_t,
float>{});
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
constexpr bool TiledPermuteN =
(QuantGroupSize::kN > 1) ? false : GemmConfig::TiledMMAPermuteN;
if(s.log_level_ > 0)
"TiledPermuteN: %d (QuantGroupSize::kN=%d)\n", TiledPermuteN, QuantGroupSize::kN);
```

**`include/ck_tile/host/tensor_shuffle_utils.hpp`**
```
auto bq_permuteN(const ck_tile::HostTensor<T>& t, index_t group_n)
ck_tile::HostTensor<T> t_view({n_ / (GemmConfig::N_Tile / group_n),
GemmConfig::N_Warp,
GemmConfig::N_Warp_Tile / group_n,
```

**`include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_flatbr_bquant_cr.hpp`**
```
index_t reg_offset = [&]() {
if constexpr(QuantGroupSize::kN >= (NWarp * WG::kN))
return (nIter * NWarp * WG::kN) / QuantGroupSize::kN * KPerBlockBQ +
return nIter * KPerBlockBQ + kQScale;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffle.cpp`**
```
using GroupSize2D8N  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 8, 128>>;
using GroupSize2D16N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 16, 128>>;
using GroupSize2D32N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 32, 128>>;
using GroupSize2D64N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 64, 128>>;
```
