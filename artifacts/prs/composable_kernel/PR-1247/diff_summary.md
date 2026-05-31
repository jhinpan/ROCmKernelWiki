# Diff summary

- **files changed:** 20
- **lines:** +2264 / -22
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp`  (+787/-0)
- `example/15_grouped_gemm/grouped_gemm_multiple_d_xdl_fp16.cpp`  (+403/-0)
- `profiler/include/profiler/profile_grouped_gemm_tile_loop_impl.hpp`  (+319/-0)
- `profiler/src/profile_grouped_gemm_tile_loop.cpp`  (+152/-0)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_tile_loop.hpp`  (+128/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp`  (+122/-4)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop.hpp`  (+108/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_f16_f16_f16_mk_nk_mn_instance.cpp`  (+77/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_f16_f16_f16_mk_kn_mn_instance.cpp`  (+75/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp`  (+15/-1)
- `include/ck/utility/sequence.hpp`  (+14/-1)
- `include/ck/utility/loop_scheduler.hpp`  (+13/-1)
- `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`  (+9/-4)
- `test/normalization_bwd_data/CMakeLists.txt`  (+9/-4)
- `test/normalization_bwd_gamma_beta/CMakeLists.txt`  (+8/-5)

## Key added lines (kernel files)

**`example/15_grouped_gemm/grouped_gemm_multiple_d_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm_tile_loop.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <index_t NumDTensor = 0>
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```

**`include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`**
```
__host__ __device__ static constexpr index_t CalculateGridSize(index_t M, index_t N)
__host__ __device__ static constexpr index_t CalculateGridSize(index_t M, index_t N)
__host__ __device__ static constexpr index_t CalculateGridSize(index_t M, index_t N)
__host__ __device__ constexpr index_t CalculateGridSize(index_t M, index_t N) const
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp`**
```
template <typename ALayout, typename BLayout, typename ELayout>
__host__ __device__ static bool
CheckTensorTransfersValidity(index_t MRaw, index_t NRaw, index_t KRaw)
const auto A_vector_dim_size = ABlockTransferSrcVectorDim == 2 ? KRaw : MRaw;
```
