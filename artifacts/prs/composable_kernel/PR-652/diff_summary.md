# Diff summary

- **files changed:** 92 (diff was byte-capped; summary is partial)
- **lines:** +3481 / -535
- **kernel-ish files:** 76

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_dl.hpp`  (+763/-0)
- `example/40_conv2d_fwd_quantization/run_conv2d_fwd_perchannel_quantization_example.inc`  (+235/-0)
- `client_example/17_grouped_gemm_fastgelu/grouped_gemm_fastgelu.cpp`  (+232/-0)
- `example/14_gemm_quantization/gemm_dl_quantization_int8.cpp`  (+204/-0)
- `client_example/09_quantization/gemm_quantization.cpp`  (+193/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/quantization/gemm_quantization.hpp`  (+168/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_fastgelu.hpp`  (+136/-0)
- `example/40_conv2d_fwd_quantization/run_conv2d_fwd_bias_relu_perchannel_quantization_example.inc`  (+5/-97)
- `example/40_conv2d_fwd_quantization/run_conv2d_fwd_bias_relu_perlayer_quantization_example.inc`  (+5/-94)
- `example/40_conv2d_fwd_quantization/run_conv2d_fwd_perlayer_quantization_example.inc`  (+4/-86)
- `example/40_conv2d_fwd_quantization/conv2d_fwd_xdl_bias_relu_perchannel_quantization_int8.cpp`  (+85/-0)
- `example/40_conv2d_fwd_quantization/conv2d_fwd_xdl_bias_relu_perlayer_quantization_int8.cpp`  (+83/-0)
- `example/40_conv2d_fwd_quantization/conv2d_fwd_xdl_perchannel_quantization_int8.cpp`  (+83/-0)
- `example/40_conv2d_fwd_quantization/conv2d_fwd_dl_bias_relu_perchannel_quantization_int8.cpp`  (+81/-0)
- `example/40_conv2d_fwd_quantization/conv2d_fwd_dl_bias_relu_perlayer_quantization_int8.cpp`  (+79/-0)

## Key added lines (kernel files)

**`client_example/09_quantization/conv2d_fwd_bias_relu_perchannel_quantization.cpp`**
```
static constexpr ck::index_t N             = 4;   // batch size
static constexpr ck::index_t K             = 64;  // output channel
static constexpr ck::index_t C             = 192; // input channel
static constexpr ck::index_t Y             = 3;   // filter H
```

**`client_example/09_quantization/conv2d_fwd_bias_relu_perlayer_quantization.cpp`**
```
static constexpr ck::index_t N             = 4;   // batch size
static constexpr ck::index_t K             = 64;  // output channel
static constexpr ck::index_t C             = 192; // input channel
static constexpr ck::index_t Y             = 3;   // filter H
```

**`client_example/09_quantization/conv2d_fwd_perchannel_quantization.cpp`**
```
static constexpr ck::index_t N             = 4;   // batch size
static constexpr ck::index_t K             = 64;  // output channel
static constexpr ck::index_t C             = 192; // input channel
static constexpr ck::index_t Y             = 3;   // filter H
```

**`client_example/09_quantization/conv2d_fwd_perlayer_quantization.cpp`**
```
static constexpr ck::index_t N             = 4;   // batch size
static constexpr ck::index_t K             = 64;  // output channel
static constexpr ck::index_t C             = 192; // input channel
static constexpr ck::index_t Y             = 3;   // filter H
```

**`client_example/09_quantization/gemm_quantization.cpp`**
```
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using AElementOp   = PassThrough;
```
