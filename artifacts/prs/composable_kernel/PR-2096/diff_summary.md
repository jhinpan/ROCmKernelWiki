# Diff summary

- **files changed:** 20 (diff was byte-capped; summary is partial)
- **lines:** +4314 / -555
- **kernel-ish files:** 16

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+1725/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal.hpp`  (+90/-509)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp`  (+542/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_xdl.inc`  (+521/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`  (+466/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp`  (+309/-0)
- `include/ck/tensor_operation/gpu/warp/wmma_gemm.hpp`  (+178/-6)
- `include/ck/utility/amd_wmma.hpp`  (+97/-1)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops.hpp`  (+85/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/CMakeLists.txt`  (+48/-22)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_wmma.inc`  (+68/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_bf16_bf16_bf16/device_gemm_wmma_universal_bf16_bf16_bf16_km_kn_mn.hpp`  (+64/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp`  (+60/-0)
- `library/src/tensor_operation_instance/gpu/CMakeLists.txt`  (+23/-13)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_bf16_bf16_bf16/device_gemm_wmma_universal_bf16_bf16_bf16_km_kn_mn_comp_default_instance.cpp`  (+25/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp`**
```
namespace ck {
template <BlockGemmPipelineVersion BlkGemmPipelineVer,
BlockGemmPipelineScheduler BlkGemmPipeSche,
index_t BlockSize,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops.hpp`**
```
namespace ck {
template <index_t BlockSize,
index_t MPerBlock,
index_t NPerBlock,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp`**
```
namespace ck {
template <index_t BlockSize,
typename ADataType,
typename BDataType,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`**
```
namespace ck {
template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
index_t BlockSize,
typename ADataType,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```
