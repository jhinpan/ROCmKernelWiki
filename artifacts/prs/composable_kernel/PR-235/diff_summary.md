# Diff summary

- **files changed:** 33
- **lines:** +770 / -743
- **kernel-ish files:** 31

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`  (+258/-0)
- `test/block_to_ctile_map/test_block_to_ctile_map.cpp`  (+100/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v3r2.hpp`  (+20/-51)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+17/-53)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v3r1.hpp`  (+19/-51)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v3r3.hpp`  (+19/-51)
- `include/ck/tensor_operation/gpu/device/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp`  (+12/-57)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4.hpp`  (+16/-52)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`  (+16/-52)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp`  (+16/-51)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_reduce_xdl_cshuffle_v1.hpp`  (+19/-46)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp`  (+19/-46)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp`  (+10/-49)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_xdl.hpp`  (+33/-11)
- `include/ck/tensor_operation/gpu/device/device_convnd_bwd_data_xdl_ndhwc_kzyxc_ndhwk.hpp`  (+23/-17)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`**
```
block_2_ctile_map_{GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_)},
if(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1_,
b_grid_desc_bk0_n_bk1_,
c_grid_desc_m_n_,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp`**
```
using Block2CTileMap = typename GridwiseGemm::DefaultBlock2CTileMap;
block_2_ctile_map_{
GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_, M01, N01)},
if(GridwiseGemm::CheckValidity(a_grid_desc_k0_m_k1_,
```

**`include/ck/tensor_operation/gpu/device/device_conv2d_backward_weight_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`**
```
block_2_ctile_map_ =
GridwiseGemm::MakeCBlockClusterAdaptor(c_grid_desc_m_n_, M01, N01, k_batch_);
block_2_ctile_map_))
arg.block_2_ctile_map_))
```

**`include/ck/tensor_operation/gpu/device/device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk.hpp`**
```
auto block_2_ctile_map =
GridwiseGemm::MakeDefaultBlock2CTileMap(descs[I2], M01, N01);
if(GridwiseGemm::CheckValidity(
descs[I0], descs[I1], descs[I2], block_2_ctile_map))
```

**`include/ck/tensor_operation/gpu/device/device_conv2d_fwd_xdl_c_shuffle_bias_activation_add_nhwc_kyxc_nhwk.hpp`**
```
block_2_ctile_map_{
GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_, M01, N01)},
if(GridwiseGemm::CheckValidity(a_grid_desc_k0_m_k1_,
b_grid_desc_k0_n_k1_,
```
