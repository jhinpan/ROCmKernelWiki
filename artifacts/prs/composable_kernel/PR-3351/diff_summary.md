# Diff summary

- **files changed:** 11
- **lines:** +485 / -51
- **kernel-ish files:** 9

## Files (by churn)

- `test/ck_tile/grouped_conv/test_ck_tile_grouped_conv_bwd_weight.cpp`  (+249/-0)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+74/-14)
- `include/ck_tile/ops/grouped_convolution/utils/split_k_utils.hpp`  (+81/-0)
- `example/ck_tile/20_grouped_convolution/run_grouped_convolution_bwd_weight_example.inc`  (+27/-24)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage_invoker.hpp`  (+11/-7)
- `include/ck_tile/host/device_prop.hpp`  (+18/-0)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_invoker.hpp`  (+10/-6)
- `test/ck_tile/grouped_conv/CMakeLists.txt`  (+7/-0)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_utils.hpp`  (+6/-0)
- `include/ck_tile/ops/grouped_convolution.hpp`  (+1/-0)
- `test/ck_tile/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_invoker.hpp`**
```
static InvokerResult grouped_conv_bwd_weight(const ck_tile::GroupedConvBwdWeightHostArgs& args,
const ck_tile::stream_config& s)
const auto kargs = Kernel::MakeKernelArgs(args);
const dim3 grids  = Kernel::GridSize(kargs);
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage_invoker.hpp`**
```
static InvokerResult grouped_conv_bwd_weight(const ck_tile::GroupedConvBwdWeightHostArgs& args,
const ck_tile::stream_config& s)
auto c_ptr       = ws_args.wei_ptr;
ws_args.wei_ptr  = ws_m_n_dev_buf.GetDeviceBuffer();
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_utils.hpp`**
```
struct InvokerResult
float ave_time;
ck_tile::index_t split_k;
```

**`example/ck_tile/20_grouped_convolution/run_grouped_convolution_bwd_weight_example.inc`**
```
InvokerResult invoke_grouped_conv_bwd_weight(ck_tile::GroupedConvBwdWeightHostArgs& args,
int n_warmup,
int n_repeat)
auto res = Invoker::template grouped_conv_bwd_weight<NDimSpatial,
```

**`include/ck_tile/host/device_prop.hpp`**
```
inline size_t get_num_cus()
hipDeviceProp_t props{};
int device;
auto status = hipGetDevice(&device);
```
