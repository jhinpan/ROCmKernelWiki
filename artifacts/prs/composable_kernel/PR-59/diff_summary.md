# Diff summary

- **files changed:** 22
- **lines:** +1282 / -279
- **kernel-ish files:** 20

## Files (by churn)

- `device_operation/include/device_gemm_xdl_splitk.hpp`  (+606/-0)
- `test/split_k/main.cpp`  (+218/-0)
- `profiler/include/profile_gemm_impl.hpp`  (+127/-77)
- `device_operation/device_gemm_xdl_splitk_f32_f32_f32_mk_nk_mn_instance.cpp`  (+56/-0)
- `device_operation/device_gemm_xdl_splitk_f32_f32_f32_km_kn_mn_instance.cpp`  (+51/-0)
- `device_operation/device_gemm_xdl_splitk_f32_f32_f32_km_nk_mn_instance.cpp`  (+51/-0)
- `device_operation/device_gemm_xdl_splitk_f32_f32_f32_mk_kn_mn_instance.cpp`  (+51/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r4.hpp`  (+31/-15)
- `device_operation/include/device_gemm_instance.hpp`  (+0/-27)
- `device_operation/include/device_gemm.hpp`  (+13/-13)
- `profiler/CMakeLists.txt`  (+14/-9)
- `profiler/profile_gemm.cpp`  (+15/-7)
- `device_operation/device_gemm_xdl_f16_f16_f16_km_kn_mn_instance.cpp`  (+5/-16)
- `device_operation/device_gemm_xdl_f16_f16_f16_km_nk_mn_instance.cpp`  (+5/-16)
- `device_operation/device_gemm_xdl_f16_f16_f16_mk_kn_mn_instance.cpp`  (+5/-16)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r4.hpp`**
```
typename AElementwiseOperation,
typename BElementwiseOperation,
typename CElementwiseOperation,
typename Block2CTileMap,
```

**`device_operation/device_gemm_xdl_f16_f16_f16_km_kn_mn_instance.cpp`**
```
using device_gemm_xdl_f16_f16_f16_km_kn_mn_instances =
void add_device_gemm_xdl_f16_f16_f16_km_kn_mn_instances(
std::vector<DeviceGemmPtr<PassThrough, PassThrough, PassThrough>>& instances)
add_device_operation_instances(instances, device_gemm_xdl_f16_f16_f16_km_kn_mn_instances{});
```

**`device_operation/device_gemm_xdl_f16_f16_f16_km_nk_mn_instance.cpp`**
```
using device_gemm_xdl_f16_f16_f16_km_nk_mn_instances =
void add_device_gemm_xdl_f16_f16_f16_km_nk_mn_instances(
std::vector<DeviceGemmPtr<PassThrough, PassThrough, PassThrough>>& instances)
add_device_operation_instances(instances, device_gemm_xdl_f16_f16_f16_km_nk_mn_instances{});
```

**`device_operation/device_gemm_xdl_f16_f16_f16_mk_kn_mn_instance.cpp`**
```
using device_gemm_xdl_f16_f16_f16_mk_kn_mn_instances =
void add_device_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(
std::vector<DeviceGemmPtr<PassThrough, PassThrough, PassThrough>>& instances)
add_device_operation_instances(instances, device_gemm_xdl_f16_f16_f16_mk_kn_mn_instances{});
```

**`device_operation/device_gemm_xdl_f16_f16_f16_mk_nk_mn_instance.cpp`**
```
using device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances =
void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(
std::vector<DeviceGemmPtr<PassThrough, PassThrough, PassThrough>>& instances)
add_device_operation_instances(instances, device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances{});
```
