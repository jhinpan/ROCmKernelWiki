# Diff summary

- **files changed:** 39
- **lines:** +1554 / -1055
- **kernel-ish files:** 36

## Files (by churn)

- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+919/-0)
- `test/ck_tile/gemm_block_scale/test_run_gemm_aquant_example.inc`  (+0/-616)
- `test/ck_tile/gemm_block_scale/test_gemm_aquant_utils.hpp`  (+0/-243)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`  (+179/-0)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+67/-20)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+50/-35)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+64/-0)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+51/-5)
- `example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`  (+34/-15)
- `test/ck_tile/epilogue/test_cshuffle_epilogue_util.hpp`  (+31/-17)
- `test/ck_tile/epilogue/test_cshuffle_epilogue.cpp`  (+41/-4)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_ut_cases.inc`  (+28/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_quant_pipeline_problem.hpp`  (+14/-13)
- `include/ck_tile/ops/gemm_group_quant.hpp`  (+0/-21)
- `include/ck_tile/ops/gemm_quant.hpp`  (+21/-0)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`**
```
using QuantGemmProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType,
BDataType,
AccDataType,
AccDataType,
```

**`example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`**
```
QuantMode == ck_tile::QuantType::RowColQuant ||
QuantMode == ck_tile::QuantType::TensorQuant,
ck_tile::GemmRowColTensorQuantPipelineProblem<typename TypeConfig::ADataType,
typename TypeConfig::BDataType,
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
.insert("quant_mode", "aquant", "Choose aquant (default), bquant, tensor or rowcol");
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
<< " QuantMode = " << quant_type_to_string(QuantMode)
else if constexpr(QuantMode == ck_tile::QuantType::RowColQuant ||
QuantMode == ck_tile::QuantType::TensorQuant)
AQK = 1; // Row quantization: tensor shape [M, 1] or [1]
```

**`include/ck_tile/core/tensor/load_tile.hpp`**
```
template <typename Tile>
concept IsLoadableTile = requires { load_tile(std::declval<Tile>()); };
```
