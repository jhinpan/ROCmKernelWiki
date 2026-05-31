# Diff summary

- **files changed:** 17
- **lines:** +232 / -77
- **kernel-ish files:** 16

## Files (by churn)

- `include/ck/utility/data_type.hpp`  (+0/-71)
- `include/ck/utility/generic_memory_space_atomic_add.hpp`  (+44/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp`  (+21/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+20/-0)
- `include/ck/tensor_operation/gpu/device/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp`  (+18/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_reduce_xdl_cshuffle_v1.hpp`  (+18/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v3r3.hpp`  (+17/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp`  (+15/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v3r2.hpp`  (+15/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp`  (+13/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4.hpp`  (+13/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`  (+13/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v3r1.hpp`  (+13/-0)
- `include/ck/utility/dynamic_buffer.hpp`  (+8/-5)
- `include/ck/utility/common_header.hpp`  (+2/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`**
```
ignore = p_a_grid;
ignore = p_b_grid;
ignore = p_c_grid;
ignore = p_d0_grid;
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp`**
```
ignore = p_a_grid;
ignore = p_b_grid;
ignore = p_c_grid;
ignore = batch_count;
```

**`include/ck/tensor_operation/gpu/device/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp`**
```
ignore = p_a_grid;
ignore = p_b_grid;
ignore = p_c_grid;
ignore = num_batches;
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_reduce_xdl_cshuffle_v1.hpp`**
```
ignore = p_a_grid;
ignore = p_b_grid;
ignore = p_c_grid;
ignore = p_d0_grid;
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp`**
```
ignore = p_a_grid;
ignore = p_b_grid;
ignore = p_c_grid;
ignore = a_element_op;
```
