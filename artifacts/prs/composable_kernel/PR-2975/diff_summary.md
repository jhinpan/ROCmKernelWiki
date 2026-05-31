# Diff summary

- **files changed:** 10
- **lines:** +135 / -87
- **kernel-ish files:** 9

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_flatbr_bquant_cr.hpp`  (+55/-45)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+22/-19)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+21/-1)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+6/-10)
- `example/ck_tile/38_block_scale_gemm/README.md`  (+12/-2)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+9/-4)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`  (+7/-0)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle_invoker.hpp`  (+1/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`  (+2/-1)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+0/-1)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`**
```
else if(data_type == "int4")
return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
ck_tile::fp8_t,
ck_tile::pk_int4_t,
```

**`example/ck_tile/03_gemm/gemm_weight_preshuffle_invoker.hpp`**
```
throw std::runtime_error("split-k is not supported yet!");
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);
if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
ck_tile::permute_vectors_i4x4_b(b_shuffle_host);
```

**`example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`**
```
int main(int argc, char* argv[]) { return !run_gemm_example<GemmConfigPreshuffleB_Bquant_prefill>(argc, argv); }
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
template <typename PrecType>
struct GemmConfigPreshuffleB_Bquant_prefill : public GemmConfigBase
static constexpr ck_tile::index_t M_Tile = 128;
static constexpr ck_tile::index_t N_Tile = 128;
```
