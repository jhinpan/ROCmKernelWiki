# Diff summary

- **files changed:** 21
- **lines:** +855 / -518
- **kernel-ish files:** 19

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`  (+354/-214)
- `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`  (+75/-16)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_instance.cpp`  (+35/-51)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp`  (+73/-1)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`  (+51/-23)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_mk_kn_mn_instance.cpp`  (+34/-34)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instance.cpp`  (+35/-21)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instance.cpp`  (+25/-24)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp`  (+21/-21)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instance.cpp`  (+25/-17)
- `include/ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp`  (+18/-18)
- `include/ck/tensor_operation/gpu/device/device_softmax.hpp`  (+16/-16)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp`  (+16/-16)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_nhwc_kyxc_nhwk.hpp`  (+16/-16)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_xdl_cshuffle.hpp`  (+16/-16)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp`**
```
struct GroupedGemmKernelArguments
__host__ __device__ GroupedGemmKernelArguments(const void* p_a_grid_,
const void* p_b_grid_,
void* p_c_grid_,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`**
```
stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, karg);
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`**
```
typename FloatA,
typename FloatB,
typename FloatC,
const index_t tile_count,
```

**`include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`**
```
__host__ __device__ constexpr index_t
CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
template <typename TopIdx>
__host__ __device__ BlockToCTileMap_3DGrid_KSplit([[maybe_unused]] TopIdx top_idx)
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp`**
```
inline std::string getPipelineVersionString(const PipelineVersion& pv)
switch(pv)
case PipelineVersion::v1: return "PipelineVersion::v1";
case PipelineVersion::v2: return "PipelineVersion::v2";
```
