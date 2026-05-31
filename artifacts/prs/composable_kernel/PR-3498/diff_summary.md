# Diff summary

- **files changed:** 25
- **lines:** +3206 / -2
- **kernel-ish files:** 24

## Files (by churn)

- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp`  (+302/-0)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp`  (+299/-0)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp`  (+296/-0)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_wmma_cshuffle_v3.hpp`  (+295/-0)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp`  (+284/-0)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp`  (+282/-0)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_wmma_cshuffle.hpp`  (+277/-0)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_dl.hpp`  (+272/-0)
- `experimental/builder/test/test_instance_string_bwd_weight_grp_conv_two_stage_wmma_v3.cpp`  (+90/-0)
- `experimental/builder/test/test_instance_string_bwd_weight_grp_conv_two_stage_xdl.cpp`  (+90/-0)
- `experimental/builder/test/test_instance_string_bwd_weight_grp_conv_wmma.cpp`  (+90/-0)
- `experimental/builder/test/test_instance_string_bwd_weight_grp_conv_wmma_v3.cpp`  (+90/-0)
- `experimental/builder/test/test_instance_string_bwd_weight_grp_conv_multiple_d_wmma_v3.cpp`  (+86/-0)
- `experimental/builder/test/test_instance_string_bwd_weight_grp_conv_xdl_v3.cpp`  (+86/-0)
- `experimental/builder/test/test_instance_string_bwd_weight_grp_conv_multiple_d_xdl.cpp`  (+84/-0)

## Key added lines (kernel files)

**`experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_dl.hpp`**
```
namespace ck::tensor_operation::device {
template <ck::index_t NDimSpatial,
typename InLayout,
typename WeiLayout,
```

**`experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_wmma_cshuffle_v3.hpp`**
```
namespace ck::tensor_operation::device {
template <ck::index_t NDimSpatial,
typename InLayout,
typename WeiLayout,
```

**`experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp`**
```
namespace ck::tensor_operation::device {
template <ck::index_t NDimSpatial,
typename InLayout,
typename WeiLayout,
```

**`experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp`**
```
namespace ck::tensor_operation::device {
template <ck::index_t NDimSpatial,
typename InLayout,
typename WeiLayout,
```

**`experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp`**
```
namespace ck::tensor_operation::device {
template <ck::index_t NDimSpatial,
typename InLayout,
typename WeiLayout,
```
