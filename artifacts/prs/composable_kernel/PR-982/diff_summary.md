# Diff summary

- **files changed:** 21
- **lines:** +518 / -855
- **kernel-ish files:** 19

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`  (+214/-354)
- `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`  (+16/-75)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_instance.cpp`  (+51/-35)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp`  (+1/-73)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`  (+23/-51)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_mk_kn_mn_instance.cpp`  (+34/-34)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instance.cpp`  (+21/-35)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instance.cpp`  (+24/-25)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp`  (+21/-21)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instance.cpp`  (+17/-25)
- `include/ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp`  (+18/-18)
- `include/ck/tensor_operation/gpu/device/device_softmax.hpp`  (+16/-16)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp`  (+16/-16)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_nhwc_kyxc_nhwk.hpp`  (+16/-16)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_xdl_cshuffle.hpp`  (+16/-16)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp`**
```
virtual void SetKBatchSize(BaseArgument* p_arg, index_t kbatch) const = 0;
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`**
```
stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, karg, b2c_map);
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`**
```
const index_t group_count)
const index_t block_id = get_block_1d_id();
index_t left     = 0;
index_t right    = group_count;
```

**`include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`**
```
__host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
__host__ __device__ BlockToCTileMap_3DGrid_KSplit() = default;
template <typename TopIdx>
__device__ constexpr auto CalculateBottomIndex(const TopIdx&) const
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`**
```
kernel_gemm_xdlops_v2r4r2_simplified(typename GridwiseGemm::Argument karg,
const Block2CTileMap& b2c_map)
ignore = b2c_map;
template <typename CGridDesc>
```
