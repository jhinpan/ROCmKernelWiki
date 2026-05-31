# Diff summary

- **files changed:** 50
- **lines:** +228 / -43
- **kernel-ish files:** 49

## Files (by churn)

- `include/ck/tensor_description/multi_index_transform.hpp`  (+32/-7)
- `include/ck/utility/dtype_vector.hpp`  (+12/-9)
- `include/ck/tensor_description/tensor_descriptor.hpp`  (+13/-4)
- `include/ck/library/utility/host_tensor.hpp`  (+9/-3)
- `include/ck_tile/core/container/tuple.hpp`  (+8/-3)
- `include/ck/utility/tuple.hpp`  (+4/-3)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp`  (+4/-2)
- `include/ck/host_utility/io.hpp`  (+3/-2)
- `include/ck/tensor_description/tensor_adaptor.hpp`  (+4/-1)
- `include/ck/utility/env.hpp`  (+4/-1)
- `include/ck/utility/static_buffer.hpp`  (+4/-1)
- `example/ck_tile/01_fmha/quant.hpp`  (+4/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp`  (+4/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_wmma.hpp`  (+4/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops_skip_b_lds.hpp`  (+4/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/bias.hpp`**
```
friend std::ostream& operator<<([[clang::lifetimebound]] std::ostream& os, const bias_info& bi)
```

**`example/ck_tile/01_fmha/mask.hpp`**
```
friend std::ostream& operator<<([[clang::lifetimebound]] std::ostream& os, const mask_info& mi)
```

**`include/ck/host_utility/io.hpp`**
```
std::ostream& operator<<([[clang::lifetimebound]] std::ostream& os, const std::vector<T>& v)
std::ostream& operator<<([[clang::lifetimebound]] std::ostream& os,
const TensorDescriptor<Ts...>& desc)
```

**`include/ck/library/utility/convolution_parameter.hpp`**
```
std::ostream& operator<<([[clang::lifetimebound]] std::ostream& os,
const ck::utils::conv::ConvParam& p);
```

**`include/ck/library/utility/host_tensor.hpp`**
```
std::ostream& LogRange([[clang::lifetimebound]] std::ostream& os, Range&& range, std::string delim)
friend std::ostream& operator<<([[clang::lifetimebound]] std::ostream& os,
const HostTensorDescriptor& desc);
friend std::ostream& operator<<([[clang::lifetimebound]] std::ostream& os, ChosenLayout tag);
```
