# Diff summary

- **files changed:** 32
- **lines:** +492 / -226
- **kernel-ish files:** 19

## Files (by churn)

- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+111/-101)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+79/-23)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+50/-5)
- `test/ck_tile/gemm/test_gemm_pipeline_kernel_types.hpp`  (+29/-14)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+37/-5)
- `test/ck_tile/gemm/test_gemm_pipeline_ut_cases.inc`  (+31/-6)
- `script/process_perf_data.sh`  (+9/-16)
- `script/process_qa_data.sh`  (+9/-16)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+19/-5)
- `script/process_perf_data.py`  (+24/-0)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+15/-8)
- `test/ck_tile/gemm/test_gemm_pipeline_compv3.cpp`  (+16/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_compv4.cpp`  (+16/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_mem.cpp`  (+16/-0)
- `example/ck_tile/03_gemm/script/run_full_test.sh`  (+5/-8)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
constexpr ck_tile::index_t M_Tile = 256;
constexpr ck_tile::index_t N_Tile = 256;
ck_tile::CShuffleEpilogueProblem<ADataType,
BDataType,
```

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
struct GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>
struct GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>
struct GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
ck_tile::CShuffleEpilogueProblem<ADataType,
BDataType,
AccDataType,
else if(tail_num == ck_tile::TailNumber::Odd)
```

**`include/ck_tile/core/arch/generic_memory_space_atomic.hpp`**
```
atomic_add(c_style_pointer_cast<bf16x2_t*>(p_dst), x.template get_as<bf16x2_t>()[I0]);
```

**`include/ck_tile/core/numeric/float8.hpp`**
```
if((x & 0xff) == 0x80)
```
