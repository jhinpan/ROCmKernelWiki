# Diff summary

- **files changed:** 23
- **lines:** +905 / -311
- **kernel-ish files:** 21

## Files (by churn)

- `test/ck_tile/gemm_block_scale/CMakeLists.txt`  (+111/-15)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant.cpp`  (+0/-99)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant.cpp`  (+0/-95)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffle.cpp`  (+0/-93)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffle_prefill_2d.cpp`  (+58/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_transpose.cpp`  (+53/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffle_decode_2d.cpp`  (+51/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_2d_small_n.cpp`  (+49/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_preshuffle.cpp`  (+48/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_2d_medium_n.cpp`  (+48/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_base_rrr_crr.cpp`  (+46/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_base_ccr.cpp`  (+42/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_base_rcr.cpp`  (+42/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_prefill.cpp`  (+41/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_1d_128.cpp`  (+41/-0)

## Key added lines (kernel files)

**`include/ck_tile/core/config.hpp`**
```
static constexpr bool CK_TILE_ARCH_GFX1011 = true;
static constexpr bool CK_TILE_ARCH_GFX1011 = false;
static constexpr bool CK_TILE_ARCH_GFX1012 = true;
static constexpr bool CK_TILE_ARCH_GFX1012 = false;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_base_ccr.cpp`**
```
using RowMajor      = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor   = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8           = ck_tile::fp8_t;
using BF8           = ck_tile::bf8_t;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_base_rcr.cpp`**
```
using RowMajor      = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor   = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8           = ck_tile::fp8_t;
using BF8           = ck_tile::bf8_t;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_base_rrr_crr.cpp`**
```
using RowMajor      = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor   = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8           = ck_tile::fp8_t;
using BF8           = ck_tile::bf8_t;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_aquant_prefill.cpp`**
```
using RowMajor      = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor   = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8           = ck_tile::fp8_t;
using BF8           = ck_tile::bf8_t;
```
