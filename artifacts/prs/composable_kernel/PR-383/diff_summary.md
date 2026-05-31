# Diff summary

- **files changed:** 17
- **lines:** +810 / -170
- **kernel-ish files:** 12

## Files (by churn)

- `example/22_cgemm/cgemm_xdl_common.hpp`  (+111/-50)
- `example/22_cgemm/cgemm_xdl_int4.cpp`  (+140/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_int4.cpp`  (+101/-0)
- `example/24_batched_gemm/batched_gemm_xdl_int4.cpp`  (+99/-0)
- `example/35_splitK_gemm/run_splitK_gemm_example.inc`  (+60/-36)
- `example/24_batched_gemm/run_batched_gemm_example.inc`  (+69/-23)
- `example/35_splitK_gemm/splitK_gemm_xdl_int4.cpp`  (+92/-0)
- `example/15_grouped_gemm/run_grouped_gemm_example.inc`  (+43/-11)
- `example/22_cgemm/cgemm_xdl_bf16.cpp`  (+11/-11)
- `example/22_cgemm/cgemm_xdl_fp16.cpp`  (+11/-11)
- `example/22_cgemm/cgemm_xdl_fp32.cpp`  (+11/-11)
- `example/22_cgemm/cgemm_xdl_int8.cpp`  (+11/-11)
- `example/22_cgemm/CMakeLists.txt`  (+10/-4)
- `example/15_grouped_gemm/CMakeLists.txt`  (+13/-0)
- `example/24_batched_gemm/CMakeLists.txt`  (+13/-0)

## Key added lines (kernel files)

**`example/15_grouped_gemm/grouped_gemm_xdl_int4.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`example/15_grouped_gemm/run_grouped_gemm_example.inc`**
```
static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
static_assert(sizeof(ADataType) == sizeof(KernelADataType));
static_assert(sizeof(BDataType) == sizeof(KernelBDataType));
static_assert(sizeof(EDataType) == sizeof(KernelEDataType));
```

**`example/22_cgemm/cgemm_xdl_bf16.cpp`**
```
return !run_cgemm_xdl<ADataType,
BDataType,
CDataType,
PassThrough,
```

**`example/22_cgemm/cgemm_xdl_common.hpp`**
```
using INT4 = ck::int4_t;
typename ReferenceCGemmInstance,
typename KernelADataType = ADataType,
typename KernelBDataType = BDataType,
```

**`example/22_cgemm/cgemm_xdl_fp16.cpp`**
```
return !run_cgemm_xdl<ADataType,
BDataType,
CDataType,
PassThrough,
```
