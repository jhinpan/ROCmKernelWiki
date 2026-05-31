# Diff summary

- **files changed:** 15
- **lines:** +433 / -270
- **kernel-ish files:** 14

## Files (by churn)

- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v3r1.hpp`  (+89/-86)
- `composable_kernel/include/tensor_operation/blockwise_gemm_xdlops.hpp`  (+54/-55)
- `composable_kernel/include/utility/debug.hpp`  (+77/-0)
- `.gitignore`  (+48/-0)
- `device_operation/include/device_gemm_xdl_c_shuffle.hpp`  (+23/-23)
- `device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp`  (+21/-22)
- `device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp`  (+24/-17)
- `device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp`  (+24/-17)
- `device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp`  (+24/-17)
- `device_operation/src/device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instance.cpp`  (+17/-17)
- `example/1_gemm_xdl/gemm_xdl.cpp`  (+15/-12)
- `host/host_tensor/include/host_tensor_generator.hpp`  (+11/-0)
- `device_operation/include/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`  (+3/-2)
- `device_operation/include/conv_utils.hpp`  (+2/-2)
- `composable_kernel/include/utility/common_header.hpp`  (+1/-0)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/blockwise_gemm_xdlops.hpp`**
```
index_t KPack>
static constexpr index_t KPerBlock =
BK0NK1BlockDesc{}.GetLength(I0) * BK0NK1BlockDesc{}.GetLength(I2);
static constexpr index_t A_K0 = AK0MK1BlockDesc{}.GetLength(I0);
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v3r1.hpp`**
```
typename AGridDesc_AK0_M_AK1,
typename BGridDesc_BK0_N_BK1,
const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
```

**`composable_kernel/include/utility/debug.hpp`**
```
namespace ck {
namespace debug {
namespace detail {
template <typename T, typename Enable = void>
```

**`device_operation/include/conv_utils.hpp`**
```
std::multiplies<std::size_t>()) *
std::multiplies<std::size_t>());
```

**`device_operation/include/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`**
```
K0PerBlock * K1,
K1, // AK1
K1, // BK1
```
