# Diff summary

- **files changed:** 25
- **lines:** +83 / -48
- **kernel-ish files:** 5

## Files (by churn)

- `example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.cpp`  (+12/-9)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/example_add_rmsnorm2d_rdquant_fwd.cpp`  (+12/-9)
- `example/ck_tile/03_gemm/CMakeLists.txt`  (+5/-2)
- `example/ck_tile/01_fmha/CMakeLists.txt`  (+4/-2)
- `example/ck_tile/10_rmsnorm2d/CMakeLists.txt`  (+4/-2)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/CMakeLists.txt`  (+4/-2)
- `example/ck_tile/09_topk_softmax/CMakeLists.txt`  (+3/-2)
- `example/ck_tile/CMakeLists.txt`  (+4/-1)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp`  (+3/-2)
- `example/CMakeLists.txt`  (+3/-1)
- `example/ck_tile/05_reduce/CMakeLists.txt`  (+3/-1)
- `example/ck_tile/17_grouped_gemm/CMakeLists.txt`  (+2/-2)
- `example/ck_tile/18_flatmm/CMakeLists.txt`  (+3/-1)
- `example/ck_tile/35_batched_transpose/CMakeLists.txt`  (+2/-2)
- `example/ck_tile/02_layernorm2d/CMakeLists.txt`  (+2/-1)

## Key added lines (kernel files)

**`example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.cpp`**
```
using ADataType        = typename TypeConfig::ADataType;
using BDataType        = typename TypeConfig::BDataType;
using GammaDataType    = typename TypeConfig::GammaDataType;
using XDataType        = typename TypeConfig::XDataType;
```

**`example/ck_tile/11_add_rmsnorm2d_rdquant/example_add_rmsnorm2d_rdquant_fwd.cpp`**
```
using ADataType        = DataType;
using BDataType        = DataType;
using GammaDataType    = DataType;
using XDataType        = DataType;
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`**
```
static constexpr bool TransposeC            = Traits::TransposeC;
```

**`include/ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp`**
```
typename CLayout_,
bool UseStructuredSparsity_ = false>
static constexpr bool UseStructuredSparsity = UseStructuredSparsity_;
```
