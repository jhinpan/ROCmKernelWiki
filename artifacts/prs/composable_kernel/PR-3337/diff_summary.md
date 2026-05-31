# Diff summary

- **files changed:** 12
- **lines:** +950 / -208
- **kernel-ish files:** 11

## Files (by churn)

- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_util_quant.hpp`  (+282/-52)
- `example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`  (+156/-77)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`  (+205/-22)
- `include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`  (+94/-15)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`  (+69/-6)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant.cpp`  (+31/-22)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`  (+48/-1)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_aquant.cpp`  (+38/-0)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_rowcol.cpp`  (+8/-5)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_tensor.cpp`  (+8/-5)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_bquant.cpp`  (+8/-3)
- `test/ck_tile/grouped_gemm_quant/CMakeLists.txt`  (+3/-0)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`**
```
template <typename GemmConfig,
typename ALayout,
typename AQLayout,
typename BLayout,
```

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`**
```
template <bool Persistent_>
static constexpr bool Persistent                = Persistent_;
template <typename PrecType, bool Persistent>
struct GemmConfigComputeV3_2 : public GemmConfigBase<Persistent>
```

**`example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`**
```
if constexpr(!GemmConfig::Persistent)
ave_time =
grouped_gemm<GemmConfig,
AQLayout,
```

**`include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`**
```
type_convert<CDataType*>(gemm_descs[i].e_ptr),
gemm_descs[i].stride_BQ,
gemm_descs[i].k_batch};
if constexpr(UsePersistentKernel)
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`**
```
constexpr index_t tail_count =
((TailNum == TailNumber::Full) || (TailNum == TailNumber::Odd)) ? 1 : 2;
} while(i < (num_loop - tail_count));
template <typename ADramBlockWindowTmp,
```
