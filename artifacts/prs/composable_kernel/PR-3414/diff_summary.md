# Diff summary

- **files changed:** 21
- **lines:** +662 / -490
- **kernel-ish files:** 18

## Files (by churn)

- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`  (+116/-320)
- `example/ck_tile/17_grouped_gemm/quant_invoke_grouped_gemm_kernel.hpp`  (+313/-0)
- `example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.hpp`  (+76/-95)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm_config.hpp`  (+2/-52)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_bquant_preshuffleb.cpp`  (+38/-0)
- `test/ck_tile/grouped_gemm_quant/CMakeLists.txt`  (+14/-11)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_aquant.cpp`  (+17/-1)
- `example/ck_tile/17_grouped_gemm/CMakeLists.txt`  (+12/-1)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_bquant.cpp`  (+8/-3)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_util_quant.hpp`  (+7/-4)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm_bf8_aquant.cpp`  (+7/-0)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm_bf8_bquant.cpp`  (+7/-0)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm_bf8_rowcol.cpp`  (+7/-0)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm_bf8_tensor.cpp`  (+7/-0)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm_fp8_aquant.cpp`  (+7/-0)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`**
```
extern template int run_gemm_example_persistency<ck_tile::fp8_t, ck_tile::QuantType::TensorQuant>(
const ck_tile::ArgParser&, std::string, std::string, bool);
extern template int run_gemm_example_persistency<ck_tile::fp8_t, ck_tile::QuantType::RowColQuant>(
const ck_tile::ArgParser&, std::string, std::string, bool);
```

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm_bf8_aquant.cpp`**
```
template int run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::AQuantGrouped>(
const ck_tile::ArgParser&, std::string, std::string, bool);
```

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm_bf8_bquant.cpp`**
```
template int run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::BQuantGrouped>(
const ck_tile::ArgParser&, std::string, std::string, bool);
```

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm_bf8_rowcol.cpp`**
```
template int run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::RowColQuant>(
const ck_tile::ArgParser&, std::string, std::string, bool);
```

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm_bf8_tensor.cpp`**
```
template int run_gemm_example_persistency<ck_tile::bf8_t, ck_tile::QuantType::TensorQuant>(
const ck_tile::ArgParser&, std::string, std::string, bool);
```
