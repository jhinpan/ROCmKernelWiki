# Diff summary

- **files changed:** 8
- **lines:** +323 / -196
- **kernel-ish files:** 7

## Files (by churn)

- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+0/-159)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant.cpp`  (+89/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant.cpp`  (+77/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffle.cpp`  (+51/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_rowcol.cpp`  (+38/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_tensor.cpp`  (+38/-0)
- `test/ck_tile/gemm_block_scale/CMakeLists.txt`  (+30/-4)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_ut_cases.inc`  (+0/-33)

## Key added lines (kernel files)

**`test/ck_tile/gemm_block_scale/test_gemm_quant_aquant.cpp`**
```
using RowMajor      = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor   = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8           = ck_tile::fp8_t;
using BF8           = ck_tile::bf8_t;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_bquant.cpp`**
```
using RowMajor      = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor   = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8           = ck_tile::fp8_t;
using BF8           = ck_tile::bf8_t;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffle.cpp`**
```
using RowMajor      = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor   = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8           = ck_tile::fp8_t;
using BF8           = ck_tile::bf8_t;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_rowcol.cpp`**
```
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8         = ck_tile::fp8_t;
using BF8         = ck_tile::bf8_t;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_tensor.cpp`**
```
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8         = ck_tile::fp8_t;
using BF8         = ck_tile::bf8_t;
```
