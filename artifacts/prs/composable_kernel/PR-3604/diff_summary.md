# Diff summary

- **files changed:** 24
- **lines:** +284 / -1060
- **kernel-ish files:** 24

## Files (by churn)

- `experimental/builder/test/validation/test_reference_execution.cpp`  (+122/-636)
- `experimental/builder/include/ck_tile/builder/factory/reference_factory.hpp`  (+76/-179)
- `experimental/builder/include/ck_tile/builder/factory/reference_common.hpp`  (+0/-118)
- `experimental/builder/include/ck_tile/builder/factory/helpers/ck_tile/conv_tile_tensor_layout.hpp`  (+26/-23)
- `experimental/builder/include/ck_tile/builder/factory/helpers/ck/conv_tensor_layout.hpp`  (+25/-22)
- `experimental/builder/include/ck_tile/builder/testing/conv_fwd_reference.hpp`  (+3/-44)
- `experimental/builder/test/unit_conv_tensor_layout.cpp`  (+16/-16)
- `experimental/builder/test/validation/test_reference_instance_traits.cpp`  (+0/-6)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_dl_factory.hpp`  (+1/-1)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_multi_d_wmma_v3_factory.hpp`  (+1/-1)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_multi_d_xdl_factory.hpp`  (+1/-1)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_two_stage_wmma_v3_factory.hpp`  (+1/-1)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_two_stage_xdl_factory.hpp`  (+1/-1)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_wmma_factory.hpp`  (+1/-1)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_wmma_v3_factory.hpp`  (+1/-1)

## Key added lines (kernel files)

**`experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_dl_factory.hpp`**
```
using Layouts                       = internal::ConvTensorLayouts<SIGNATURE>;
```

**`experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_multi_d_wmma_v3_factory.hpp`**
```
using Layouts                       = internal::ConvTensorLayouts<SIGNATURE>;
```

**`experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_multi_d_xdl_factory.hpp`**
```
using Layouts                       = internal::ConvTensorLayouts<SIGNATURE>;
```

**`experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_two_stage_wmma_v3_factory.hpp`**
```
using Layouts                       = internal::ConvTensorLayouts<SIGNATURE>;
```

**`experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_two_stage_xdl_factory.hpp`**
```
using Layouts                       = internal::ConvTensorLayouts<SIGNATURE>;
```
