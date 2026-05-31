# Diff summary

- **files changed:** 11
- **lines:** +828 / -8
- **kernel-ish files:** 10

## Files (by churn)

- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+249/-0)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+221/-2)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+178/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_mem_decode_interwave.cpp`  (+41/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_mem_decode_intrawave.cpp`  (+41/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_mem_prefill_interwave.cpp`  (+41/-0)
- `test/ck_tile/gemm_block_scale/CMakeLists.txt`  (+29/-1)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+23/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_prefill.cpp`  (+3/-3)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp`  (+1/-1)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_mem.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp`**
```
using GemmConfig = GemmConfigQuantDecodeInterwave<T>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
template <typename PrecType>
struct GemmConfigQuantDecodeInterwave : public GemmConfigBase
static constexpr ck_tile::index_t M_Tile = 16;
static constexpr ck_tile::index_t N_Tile = 64;
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
ck_tile::FillConstant<AQDataType>{static_cast<AQDataType>(1.0f)}(*aq_tensor_ptr);
ck_tile::FillConstant<BDataType>{static_cast<BDataType>(0x38)}(b_k_n);
if constexpr(QuantMode == ck_tile::QuantType::RowColQuant)
ck_tile::FillConstant<BQDataType>{static_cast<BQDataType>(0.5f)}(*bq_tensor_ptr);
```

**`include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`**
```
template <typename GemmTraits>
struct BlockGemmImpl<GemmPipelineScheduler::Interwave, GemmTraits>
static constexpr index_t KPerThread     = GemmTraits::KPerThread;
static constexpr index_t NumMacClusters = GemmTraits::InterWaveSchedulingMacClusters;
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_mem.hpp`**
```
[](const BDataType& a) { return a; },
```
