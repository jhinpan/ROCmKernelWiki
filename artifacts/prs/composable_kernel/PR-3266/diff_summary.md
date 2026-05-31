# Diff summary

- **files changed:** 40
- **lines:** +198 / -195
- **kernel-ish files:** 40

## Files (by churn)

- `codegen/include/ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_op.hpp`  (+60/-60)
- `codegen/include/ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_problem.hpp`  (+56/-56)
- `codegen/src/device_grouped_conv_fwd_multiple_abd.cpp`  (+41/-42)
- `codegen/src/device_batched_gemm_softmax_gemm.cpp`  (+1/-2)
- `codegen/src/device_gemm_multiple_d.cpp`  (+1/-2)
- `codegen/test/batched_gemm_softmax_gemm.cpp`  (+3/-0)
- `codegen/test/include/test.hpp`  (+3/-0)
- `codegen/driver/main.cpp`  (+1/-1)
- `codegen/include/ck/host/device_batched_gemm_softmax_gemm/operation.hpp`  (+1/-1)
- `codegen/include/ck/host/device_batched_gemm_softmax_gemm/problem.hpp`  (+1/-1)
- `codegen/include/ck/host/device_gemm_multiple_d.hpp`  (+1/-1)
- `codegen/include/ck/host/device_gemm_multiple_d/operation.hpp`  (+1/-1)
- `codegen/include/ck/host/device_gemm_multiple_d/problem.hpp`  (+1/-1)
- `codegen/include/ck/host/headers.hpp`  (+1/-1)
- `codegen/include/ck/host/operation/gemm.hpp`  (+1/-1)

## Key added lines (kernel files)

**`codegen/include/ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_op.hpp`**
```
namespace ck {
namespace host {
namespace conv {
struct Operation_Conv_Fwd_Xdl_Cshuffle
```

**`codegen/include/ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_problem.hpp`**
```
namespace ck {
namespace host {
namespace conv {
struct Problem_Conv_Fwd
```

**`codegen/src/device_grouped_conv_fwd_multiple_abd.cpp`**
```
namespace ck {
namespace host {
namespace conv {
std::string Problem_Conv_Fwd::GetIncludeHeader() const
```
