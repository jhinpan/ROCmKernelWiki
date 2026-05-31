# Diff summary

- **files changed:** 29
- **lines:** +2555 / -489
- **kernel-ish files:** 28

## Files (by churn)

- `experimental/builder/test/conv/ck/test_conv_traits.cpp`  (+1102/-0)
- `experimental/builder/include/ck_tile/builder/reflect/conv_traits_helpers.hpp`  (+225/-140)
- `experimental/builder/test/test_conv_description.cpp`  (+238/-20)
- `experimental/builder/include/ck_tile/builder/reflect/conv_description.hpp`  (+97/-104)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_xdl_cshuffle.hpp`  (+57/-31)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_wmma_cshuffle_v3.hpp`  (+48/-23)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp`  (+50/-21)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_wmma_cshuffle.hpp`  (+47/-24)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp`  (+48/-23)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp`  (+49/-21)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp`  (+47/-21)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp`  (+45/-21)
- `experimental/builder/include/ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp`  (+57/-0)
- `experimental/builder/include/ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_xdl_cshuffle.hpp`  (+56/-0)
- `experimental/builder/include/ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp`  (+53/-0)

## Key added lines (kernel files)

**`experimental/builder/include/ck_tile/builder/reflect/conv_describe.hpp`**
```
traits, []<typename T = Instance>() { return reflect::instance_string<T>(); });
```

**`experimental/builder/include/ck_tile/builder/reflect/conv_description.hpp`**
```
ConvDescription(ConvTraits traits, std::function<std::string()> instance_string_getter)
: traits_(std::move(traits)), instance_string_getter_(std::move(instance_string_getter))
oss << traits_.spatial_dim << "D " << traits_.direction << " convolution";
f.writeLine(0, traits_.spatial_dim, "D ", traits_.direction, " Convolution Kernel");
```

**`experimental/builder/include/ck_tile/builder/reflect/conv_traits.hpp`**
```
std::optional<builder::GemmPadding> gemm_padding = std::nullopt;
std::optional<int> num_gemm_k_prefetch_stage = std::nullopt;
std::optional<int> max_transpose_transfer_src_scalar_per_vector = std::nullopt;
std::optional<int> max_transpose_dst_scalar_per_vector          = std::nullopt;
```

**`experimental/builder/include/ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_multiple_d_wmma_cshuffle_v3.hpp`**
```
namespace ck_tile::reflect::conv {
template <typename Instance>
requires HasInstanceTraits<Instance> &&
std::same_as<typename InstanceTraits<Instance>::device_kernel_tag,
```

**`experimental/builder/include/ck_tile/builder/reflect/conv_traits_device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp`**
```
namespace ck_tile::reflect::conv {
template <typename Instance>
requires HasInstanceTraits<Instance> &&
std::same_as<typename InstanceTraits<Instance>::device_kernel_tag,
```
