# Diff summary

- **files changed:** 11
- **lines:** +554 / -194
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+227/-7)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+125/-47)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+84/-7)
- `example/ck_tile/18_flatmm/run_flatmm_example.inc`  (+46/-33)
- `include/ck_tile/ops/flatmm/block/block_flatmm_asmem_bsmem_creg_v1.hpp`  (+6/-71)
- `example/ck_tile/18_flatmm/flatmm_basic.hpp`  (+44/-8)
- `include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`  (+9/-19)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+8/-0)
- `example/ck_tile/18_flatmm/CMakeLists.txt`  (+2/-1)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`  (+1/-1)
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/18_flatmm/flatmm_basic.cpp`**
```
template <typename ADataType,
typename BDataType,
typename AccDataType,
typename CDataType,
```

**`example/ck_tile/18_flatmm/flatmm_basic.hpp`**
```
template <typename ADataType, typename BDataType = ADataType, typename CDataType = ADataType>
template <>
struct GemmBasicTypeConfig<ck_tile::bf16_t>
using ADataType   = ck_tile::bf16_t;
```

**`example/ck_tile/18_flatmm/run_flatmm_example.inc`**
```
template <typename T>
constexpr const char* DataTypeToString() {
if constexpr (std::is_same_v<T, ck_tile::half_t>) {
return "fp16";
```

**`include/ck_tile/ops/flatmm/block/block_flatmm_asmem_bsmem_creg_v1.hpp`**
```
template <typename CBlockTensor, typename ABlockWindow, typename BFlatBlockTensor>
ABlockWindow& a_warp_windows,
BFlatBlockTensor& b_warp_tensor) const
constexpr index_t MPerBlock = BlockGemmShape::kM;
```

**`include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`**
```
return make_naive_tensor_view<address_space_enum::global>(
return make_naive_tensor_view<address_space_enum::global>(
MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
a_ptr, b_flat_ptr, c_ptr, kargs, splitk_batch_offset);
```
