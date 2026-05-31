# Diff summary

- **files changed:** 55 (diff was byte-capped; summary is partial)
- **lines:** +2123 / -1562
- **kernel-ish files:** 53

## Files (by churn)

- `device_operation/include/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp`  (+676/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_dlops_v3.hpp`  (+20/-346)
- `example/10_conv3d_fwd_xdl/conv3d_fwd_xdl.cpp`  (+281/-0)
- `device_operation/include/device_conv3d_fwd_naive_ndhwc_kzyxc_ndhwk.hpp`  (+276/-0)
- `composable_kernel/include/problem_transform/transform_forward_convolution3d_into_gemm_v4r4r4_ndhwc_kzyxc_ndhwk.hpp`  (+150/-0)
- `host/driver_offline/include/driver_gemm_dlops_v1r2.hpp`  (+7/-142)
- `host/driver_offline/include/driver_convolution_add_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp`  (+4/-140)
- `host/driver_offline/include/driver_convolution_maxpool_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp`  (+4/-133)
- `host/driver_offline/include/driver_gemm_dlops_v1r3.hpp`  (+7/-125)
- `device_operation_reference/include/naive_conv_fwd.hpp`  (+122/-0)
- `host/driver_offline/include/driver_convolution_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp`  (+4/-118)
- `composable_kernel/include/tensor_description/multi_index_transform.hpp`  (+87/-0)
- `composable_kernel/include/tensor_operation/gridwise_batched_gemm_xdlops_v2r3.hpp`  (+7/-66)
- `device_operation/include/convolution_utility.hpp`  (+73/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`  (+7/-65)

## Key added lines (kernel files)

**`composable_kernel/include/config.hpp`**
```
using index_t      = int32_t;
using long_index_t = int64_t;
```

**`composable_kernel/include/problem_transform/transform_forward_convolution3d_into_gemm_v4r4r4_ndhwc_kzyxc_ndhwk.hpp`**
```
namespace ck {
template <typename... In,
typename... Wei,
typename... Out,
```

**`composable_kernel/include/tensor_description/multi_index_transform.hpp`**
```
template <typename Modulus, typename UpLength>
struct Modulo
using LowerIndex = MultiIndex<1>;
using UpperIndex = MultiIndex<1>;
```

**`composable_kernel/include/tensor_description/multi_index_transform_helper.hpp`**
```
template <typename UpperIndex>
__host__ __device__ constexpr auto make_insert_transform(const UpperIndex& up_idx)
return Insert<UpperIndex>{up_idx};
template <typename Modulus, typename UpLength>
```

**`composable_kernel/include/tensor_operation/gridwise_batched_gemm_xdlops_v2r3.hpp`**
```
MakeDefaultBlock2CTileMap(const CGridDesc_G_M_N& c_grid_desc_g_m_n, index_t M01, index_t N01)
const auto cblockid_to_g_m00_m01_n00_n01_block_cluster_adaptor =
const auto cblockid_to_g_m0_n0_block_cluster_adaptor =
cblockid_to_g_m00_m01_n00_n01_block_cluster_adaptor);
```
