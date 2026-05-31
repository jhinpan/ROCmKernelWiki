# Diff summary

- **files changed:** 37 (diff was byte-capped; summary is partial)
- **lines:** +3779 / -203
- **kernel-ish files:** 32

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_reduce_xdl_cshuffle_v1.hpp`  (+892/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_reduce_xdl_cshuffle.hpp`  (+746/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp`  (+684/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_xdl_cshuffle.hpp`  (+644/-0)
- `example/16_gemm_reduce/gemm_reduce_xdl_fp16.cpp`  (+269/-0)
- `library/include/ck/library/host_tensor/host_tensor.hpp`  (+37/-42)
- `example/01_gemm/gemm_xdl_fp16.cpp`  (+23/-46)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_kn_mn_instance.cpp`  (+68/-0)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_nk_mn_instance.cpp`  (+68/-0)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_kn_mn_instance.cpp`  (+68/-0)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_nk_mn_instance.cpp`  (+65/-0)
- `include/ck/tensor_operation/gpu/device/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp`  (+22/-35)
- `include/ck/tensor_operation/gpu/device/device_gemm_reduce.hpp`  (+49/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_bias.hpp`  (+40/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm.hpp`  (+1/-33)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_fp16.cpp`**
```
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle
<     Row,     Col,     Row,  F16,   F16,   F16,      F32,      F32,  AElementOp,  BElementOp,  CElementOp,    GemmDefau
invoker.Run(argument);
KernelTimer timer;
```

**`example/16_gemm_reduce/gemm_reduce_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`include/ck/config.hpp`**
```
enum struct InMemoryDataOperationEnum_t
enum struct ActivTypeEnum_t
```

**`include/ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp`**
```
enum struct ConvolutionBackwardDataSpecialization_t
```

**`include/ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp`**
```
enum struct ConvolutionForwardSpecialization_t
case ConvolutionForwardSpecialization_t::Default: return "Default";
case ConvolutionForwardSpecialization_t::Filter1x1Pad0: return "Filter1x1Pad0";
case ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0: return "Filter1x1Stride1Pad0";
```
