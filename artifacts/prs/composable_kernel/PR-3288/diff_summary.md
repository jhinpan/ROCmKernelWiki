# Diff summary

- **files changed:** 5 (diff was byte-capped; summary is partial)
- **lines:** +86 / -795
- **kernel-ish files:** 5

## Files (by churn)

- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp`  (+60/-373)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_comp_instance.hpp`  (+0/-159)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_mem_instance.hpp`  (+0/-131)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_merged_groups_instance.hpp`  (+0/-125)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_ab_instance.hpp`  (+26/-7)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp`**
```
DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,  BF16,  BF16,     
DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,  BF16,  BF16,     
DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,  BF16,  BF16,     
DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,  BF16,  BF16,     
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_ab_instance.hpp`**
```
DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, ck::Tuple<>, ELayout, ck::Tuple<BF16,  B
DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, ck::Tuple<>, ELayout, ck::Tuple<BF16,  B
DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, ck::Tuple<>, ELayout, ck::Tuple<BF16,  B
DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout, ck::Tuple<>, ELayout, ck::Tuple<BF16,  B
```
