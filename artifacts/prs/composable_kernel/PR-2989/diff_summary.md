# Diff summary

- **files changed:** 23
- **lines:** +2678 / -332
- **kernel-ish files:** 21

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_layernorm_wmma_cshuffle_v3.hpp`  (+896/-0)
- `include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_welford_wmma.hpp`  (+510/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+64/-267)
- `include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma_base.hpp`  (+253/-0)
- `include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma.hpp`  (+195/-0)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7r3.hpp`  (+118/-15)
- `library/src/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm/device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_km_kn_mn_mn_mn_instance.cpp`  (+108/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm/device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_km_nk_mn_mn_mn_instance.cpp`  (+108/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm/device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_mk_kn_mn_mn_mn_instance.cpp`  (+108/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm/device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_instance.cpp`  (+105/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm.hpp`  (+93/-1)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3.hpp`  (+39/-7)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+15/-9)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_b_scale.hpp`  (+15/-9)
- `profiler/include/profiler/profile_gemm_add_relu_add_layernorm_impl.hpp`  (+10/-6)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp`**
```
static constexpr auto MAccVgprs =
wmma_gemm.GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths()[I2];
```

**`include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3.hpp`**
```
index_t NumThreadScratch = 1,
typename InterDatas      = DstDatas>
if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() ||
if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() ||
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3.hpp`**
```
constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<
typename GridwiseGemm::EpilogueCShuffle>();
__shared__ char p_shared[LDS_size];
auto epilogue_args = typename GridwiseGemm::EpilogueCShuffle{};
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3_b_scale.hpp`**
```
constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<
typename GridwiseGemm::EpilogueCShuffle>();
__shared__ char p_shared[LDS_size];
auto epilogue_args = typename GridwiseGemm::EpilogueCShuffle{};
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_layernorm_wmma_cshuffle_v3.hpp`**
```
namespace ck {
template <typename GridwiseGemm,
typename EMeanVarDataType,
bool HasMainKBlockLoop,
```
