# Diff summary

- **files changed:** 14
- **lines:** +1160 / -664
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_xdl.hpp`  (+430/-258)
- `example/28_grouped_gemm_bias/grouped_gemm_bias_xdl_fp16.cpp`  (+278/-0)
- `test/grouped_gemm/grouped_gemm_fp16.cpp`  (+26/-174)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm.hpp`  (+134/-0)
- `profiler/include/profile_grouped_gemm_impl.hpp`  (+44/-79)
- `example/15_grouped_gemm/grouped_gemm_xdl_fp16.cpp`  (+51/-41)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm.hpp`  (+69/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_mk_nk_mn_instance.cpp`  (+30/-34)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_mk_kn_mn_instance.cpp`  (+32/-22)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_km_nk_mn_instance.cpp`  (+32/-14)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_km_kn_mn_instance.cpp`  (+32/-13)
- `include/ck/tensor_operation/gpu/device/device_gemm.hpp`  (+0/-29)
- `example/28_grouped_gemm_bias/CMakeLists.txt`  (+1/-0)
- `example/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/15_grouped_gemm/grouped_gemm_xdl_fp16.cpp`**
```
using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F16;
```

**`example/28_grouped_gemm_bias/grouped_gemm_bias_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
struct GemmDesc
```

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm_xdl.hpp`**
```
typename CDEElementwiseOperation,
kernel_grouped_gemm_xdl(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
const index_t group_count,
const AElementwiseOperation a_element_op,
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
