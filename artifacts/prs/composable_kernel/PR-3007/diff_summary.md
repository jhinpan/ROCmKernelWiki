# Diff summary

- **files changed:** 11
- **lines:** +671 / -68
- **kernel-ish files:** 9

## Files (by churn)

- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_util_quant.hpp`  (+441/-0)
- `example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`  (+101/-31)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant.cpp`  (+49/-0)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`  (+16/-21)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`  (+16/-14)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_ut_cases.inc`  (+28/-0)
- `test/ck_tile/grouped_gemm_quant/CMakeLists.txt`  (+10/-0)
- `include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`  (+7/-0)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+1/-1)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_multi_d_example.inc`  (+1/-1)
- `test/ck_tile/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`**
```
typename CDataType,
ck_tile::QuantType QuantMode>
using GemmUniversalTraits = ck_tile::TileGemmQuantTraits<GemmConfig::kPadM,
GemmConfig::kPadN,
```

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`**
```
"stride_As",
"Tensor A strides - it is empty by default.") // stride_As/stride_Bs/stride_Cs/stride_AQs/stride_BQs
.insert("kbatch", "1", "kbatch for SplitK")
.insert("quant_mode", "tensor", "Choose tensor (default), or rowcol");
```

**`example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`**
```
ck_tile::QuantType QuantMode,
CDataType,
QuantMode>(stream, group_count, kargs_ptr);
std::string op_name = "Quant Grouped Gemm (" + ck_tile::quant_type_to_string(QuantMode) + ")";
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`**
```
return group_count != 0 && ((args.size() == static_cast<size_t>(group_count)) && ...);
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_multi_d_example.inc`**
```
return group_count != 0 && ((args.size() == static_cast<size_t>(group_count)) && ...);
```
