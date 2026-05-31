# Diff summary

- **files changed:** 24 (diff was byte-capped; summary is partial)
- **lines:** +1903 / -4
- **kernel-ish files:** 18

## Files (by churn)

- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_outelementop_instance.hpp`  (+275/-0)
- `example/62_convnd_activ/dynamic_unary/convnd_fwd_activ_dynamic_unary_wmma_common.hpp`  (+245/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_dynamic_op_instance.hpp`  (+143/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_scaleadd_relu_instance.hpp`  (+142/-0)
- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+132/-0)
- `example/62_convnd_activ/convscale_add/convnd_fwd_wmma_convscale_add_fp8.cpp`  (+99/-0)
- `example/62_convnd_activ/convinvscale/convnd_fwd_wmma_convinvscale_fp8.cpp`  (+98/-0)
- `example/62_convnd_activ/convscale/convnd_fwd_wmma_convscale_bf8.cpp`  (+98/-0)
- `example/62_convnd_activ/convscale/convnd_fwd_wmma_convscale_bf8_fp8.cpp`  (+98/-0)
- `example/62_convnd_activ/convscale/convnd_fwd_wmma_convscale_fp8.cpp`  (+98/-0)
- `example/62_convnd_activ/convscale/convnd_fwd_wmma_convscale_fp8_bf8.cpp`  (+98/-0)
- `example/62_convnd_activ/convscale_relu/convnd_fwd_wmma_convscale_relu_fp8.cpp`  (+98/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_binary_outelementop_instance.hpp`  (+95/-0)
- `example/62_convnd_activ/convscale_reduce/convnd_fwd_wmma_convscale_amax_fp8.cpp`  (+94/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convinvscale.hpp`  (+17/-0)

## Key added lines (kernel files)

**`example/62_convnd_activ/convinvscale/convnd_fwd_wmma_convinvscale_fp8.cpp`**
```
using InDataType       = ck::f8_t;
using WeiDataType      = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = float;
```

**`example/62_convnd_activ/convscale/convnd_fwd_wmma_convscale_bf8.cpp`**
```
using InDataType       = ck::bf8_t;
using WeiDataType      = ck::bf8_t;
using AccDataType      = float;
using CShuffleDataType = float;
```

**`example/62_convnd_activ/convscale/convnd_fwd_wmma_convscale_bf8_fp8.cpp`**
```
using InDataType       = ck::bf8_t;
using WeiDataType      = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = float;
```

**`example/62_convnd_activ/convscale/convnd_fwd_wmma_convscale_fp8.cpp`**
```
using InDataType       = ck::f8_t;
using WeiDataType      = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = float;
```

**`example/62_convnd_activ/convscale/convnd_fwd_wmma_convscale_fp8_bf8.cpp`**
```
using InDataType       = ck::f8_t;
using WeiDataType      = ck::bf8_t;
using AccDataType      = float;
using CShuffleDataType = float;
```
