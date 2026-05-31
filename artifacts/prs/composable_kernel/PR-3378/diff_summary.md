# Diff summary

- **files changed:** 5 (diff was byte-capped; summary is partial)
- **lines:** +141 / -500
- **kernel-ish files:** 5

## Files (by churn)

- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp`  (+22/-288)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_exp_gemm_wmma_universal_km_kn_mn_instance.hpp`  (+67/-72)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_wmma_instance.hpp`  (+0/-83)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_wmma_bilinear_instance.hpp`  (+26/-37)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_wmma_instance.hpp`  (+26/-20)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_exp_gemm_wmma_universal_km_kn_mn_instance.hpp`**
```
template <typename InOutDataType>
using device_gemm_wmma_universal_km_kn_mn_GemmDefault_instances = std::tuple<
DeviceBatchedGemmMultiD_Wmma_CShuffleV3<     Col,     Row,   Tuple<>,    Row, InOutDataType, InOutDataType,    Tuple<>, 
DeviceBatchedGemmMultiD_Wmma_CShuffleV3<     Col,     Row,   Tuple<>,    Row, InOutDataType, InOutDataType,    Tuple<>, 
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp`**
```
DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F1
DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF1
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_wmma_instance.hpp`**
```
BlockGemmPipelineScheduler Scheduler     = BlockGemmPipelineScheduler::Intrawave,
BlockGemmPipelineVersion PipelineVersion = BlockGemmPipelineVersion::v1>
DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F
DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_wmma_bilinear_instance.hpp`**
```
DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout, Tuple<BLayout>,    F16
DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout, Tuple<BLayout>,    F16
DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout, Tuple<BLayout>,    F16
DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout, Tuple<BLayout>,    F16
```
