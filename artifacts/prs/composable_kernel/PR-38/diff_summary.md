# Diff summary

- **files changed:** 28
- **lines:** +3642 / -529
- **kernel-ish files:** 26

## Files (by churn)

- `host/driver_offline/include/device_convolution_backward_data_implicit_gemm_v4r1r2_xdlops_nhwc_kyxc_nhwk_1x1.hpp`  (+389/-0)
- `host/driver_offline/include/device_gemm_xdlops_mk_nk_mn.hpp`  (+337/-48)
- `host/driver_offline/include/device_gemm_xdlops_mk_nk_nm.hpp`  (+347/-0)
- `host/driver_offline/include/device_gemm_xdlops_mk_kn_mn.hpp`  (+290/-46)
- `host/driver_offline/include/device_gemm_xdlops_km_kn_mn.hpp`  (+289/-45)
- `host/driver_offline/include/device_gemm_xdlops_km_nk_mn.hpp`  (+289/-45)
- `host/driver_offline/include/device_gemm_xdlops_mk_kn_nm.hpp`  (+291/-0)
- `host/driver_offline/include/device_convolution_backward_data_implicit_gemm_v4r1r2_xdlops_nhwc_kyxc_nhwk.hpp`  (+193/-78)
- `host/driver_offline/include/device_gemm_xdlops_km_kn_nm.hpp`  (+263/-0)
- `host/driver_offline/include/device_gemm_xdlops_km_nk_nm.hpp`  (+263/-0)
- `host/driver_offline/src/gemm_driver_offline.cpp`  (+90/-96)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`  (+131/-40)
- `script/run.sh`  (+115/-36)
- `composable_kernel/include/problem_transform/transform_backward_data_convolution_into_gemm_v4r1r2_nhwc_kyxc_nhwk.hpp`  (+90/-13)
- `host/driver_offline/include/device_convolution_forward_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk.hpp`  (+68/-8)

## Key added lines (kernel files)

**`composable_kernel/include/problem_transform/transform_backward_data_convolution_into_gemm_v4r1r2_nhwc_kyxc_nhwk.hpp`**
```
typename IYTilda,
typename IXTilda,
IYTilda i_ytilda,
IXTilda i_xtilda,
```

**`composable_kernel/include/tensor_description/multi_index_transform_helper.hpp`**
```
template <typename LowLength, typename RightPadLength, bool SkipIsValidCheck = false>
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`**
```
const CM0N0M1N1M2M3M4N2GridDesc c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
bool CAccessOrderMRepeatNRepeat,
bool ABlockLdsExtraM,
bool BBlockLdsExtraN>
```

**`host/driver_offline/include/debug.hpp`**
```
namespace debug {
namespace debug_driver_gemm_xdlops_v2r3 {
static ck::index_t M01 = 1;
static ck::index_t N01 = 1;
```

**`host/driver_offline/include/device_convolution_backward_data_implicit_gemm_v4r1_xdlops_nhwc_kyxc_nhwk.hpp`**
```
false,  // CAccessOrderMRepeatNRepeat
false,  // ABlockLdsExtraM
false   // BBlockLdsExtraN
debug_driver_gemm_xdlops_v2r3::M01,
```
