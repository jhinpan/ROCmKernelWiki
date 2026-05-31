# Diff summary

- **files changed:** 37
- **lines:** +1726 / -612
- **kernel-ish files:** 34

## Files (by churn)

- `experimental/builder/test/unit_conv_tensor_layout.cpp`  (+381/-16)
- `experimental/builder/include/ck_tile/builder/factory/helpers/conv_tensor_layout.hpp`  (+171/-84)
- `experimental/builder/include/ck_tile/builder/README.md`  (+244/-0)
- `experimental/builder/include/ck_tile/builder/types.hpp`  (+130/-98)
- `experimental/builder/include/ck_tile/builder/factory/helpers/conv_tensor_type.hpp`  (+151/-61)
- `experimental/builder/include/ck_tile/builder/conv_signature_concepts.hpp`  (+129/-20)
- `experimental/builder/test/test_conv_description.cpp`  (+60/-19)
- `experimental/builder/include/ck_tile/builder/reflect/conv_traits.hpp`  (+45/-26)
- `experimental/builder/include/ck_tile/builder/factory/helpers/conv_elementwise_op.hpp`  (+52/-14)
- `experimental/builder/test/unit_conv_tensor_type.cpp`  (+16/-45)
- `experimental/builder/include/ck_tile/builder/conv_signature_utils.hpp`  (+0/-47)
- `experimental/builder/test/conv/test_ckb_conv_fwd_2d_bf16_scaleadd_relu.cpp`  (+46/-0)
- `experimental/builder/test/conv/test_ckb_conv_fwd_2d_dl_fp16.cpp`  (+26/-16)
- `experimental/builder/test/conv/test_ckb_conv_fwd_2d_bf16.cpp`  (+24/-16)
- `experimental/builder/test/conv/test_ckb_conv_fwd_2d_large_tensor_fp16.cpp`  (+24/-16)

## Key added lines (kernel files)

**`experimental/builder/include/ck_tile/builder/conv_signature_concepts.hpp`**
```
concept ValidConvDataType =
(T == DataType::FP32) || (T == DataType::FP16) || (T == DataType::BF16) ||
(T == DataType::FP8) || (T == DataType::I8) || (T == DataType::U8);
template <TensorLayout L>
```

**`experimental/builder/include/ck_tile/builder/factory/conv_fwd_dl_factory.hpp`**
```
using Layouts = internal::ConvTensorLayouts<SIGNATURE, SPATIAL_DIM, ConvDirection::FORWARD>;
using Types   = internal::FwdConvTensorDataTypes<SIGNATURE>;
using Ops     = internal::ElementwiseOps<SIGNATURE>;
```

**`experimental/builder/include/ck_tile/builder/factory/conv_fwd_large_tensor_factory.hpp`**
```
using Layouts = internal::ConvTensorLayouts<SIGNATURE, SPATIAL_DIM, ConvDirection::FORWARD>;
using Types   = internal::FwdConvTensorDataTypes<SIGNATURE>;
using Ops     = internal::ElementwiseOps<SIGNATURE>;
```

**`experimental/builder/include/ck_tile/builder/factory/conv_fwd_v3_factory.hpp`**
```
using Layouts = internal::ConvTensorLayouts<SIGNATURE, SPATIAL_DIM, ConvDirection::FORWARD>;
using Types   = internal::FwdConvTensorDataTypes<SIGNATURE>;
using Ops     = internal::ElementwiseOps<SIGNATURE>;
```

**`experimental/builder/include/ck_tile/builder/factory/conv_fwd_wmma_factory.hpp`**
```
using Layouts = internal::ConvTensorLayouts<SIGNATURE, SPATIAL_DIM, ConvDirection::FORWARD>;
using Types   = internal::FwdConvTensorDataTypes<SIGNATURE>;
using Ops     = internal::ElementwiseOps<SIGNATURE>;
```
