# Diff summary

- **files changed:** 38
- **lines:** +349 / -1885
- **kernel-ish files:** 37

## Files (by churn)

- `test/ck_tile/gemm/test_gemm_pipeline_smoke_util.hpp`  (+0/-450)
- `test/ck_tile/gemm/test_gemm_pipeline_smoke_run_test.inc`  (+0/-392)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_run_test.inc`  (+0/-260)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_run_test.inc`  (+0/-218)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+46/-84)
- `include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`  (+23/-71)
- `test/ck_tile/gemm/test_gemm_pipeline_kernel_types.hpp`  (+82/-6)
- `test/ck_tile/gemm/test_gemm_pipeline_type_param_product.hpp`  (+0/-63)
- `test/ck_tile/gemm/CMakeLists.txt`  (+10/-44)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+30/-8)
- `test/ck_tile/gemm/test_gemm_pipeline_wmma_base.hpp`  (+24/-13)
- `test/ck_tile/gemm/test_gemm_pipeline_ut_cases.inc`  (+34/-1)
- `include/ck_tile/ops/common/load_interleaved_pk_type.hpp`  (+13/-16)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_cases.hpp`  (+0/-25)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_cases.hpp`  (+0/-25)

## Key added lines (kernel files)

**`include/ck_tile/core/tensor/tile_window.hpp`**
```
CK_TILE_DEVICE void load(DistributedTensor& dst_tensor,
CK_TILE_DEVICE void load(DistributedTensor& dst_tensor,
CK_TILE_DEVICE void async_load_raw(LdsTileWindow_&& lds_tile,
CK_TILE_DEVICE void async_load_with_offset(index_t offset,
```

**`include/ck_tile/ops/common/load_interleaved_pk_type.hpp`**
```
template <typename DstDataType, index_t UnaryOpSize>
using DstVectorType = DstDataType __attribute__((ext_vector_type(UnaryOpSize)));
elementwise_op(warp_tile.get_thread_buffer().template get_as<DstVectorType>()(i),
template <typename SrcDataType,
```

**`include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`**
```
using ATypeToUse =
std::conditional_t<std::is_same_v<ADataType, pk_int4_t>, BDataType, ADataType>;
using BTypeToUse =
std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`**
```
static constexpr bool is_a_load_tr = []() {
if constexpr(std::is_same_v<BDataType, pk_int4_t>)
return false;
return std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
```

**`include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`**
```
static constexpr bool is_a_load_tr = []() {
using BDataType = remove_cvref_t<typename Problem::BDataType>;
if constexpr(std::is_same_v<BDataType, pk_int4_t>)
return false;
```
