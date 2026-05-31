# Diff summary

- **files changed:** 24 (diff was byte-capped; summary is partial)
- **lines:** +1079 / -348
- **kernel-ish files:** 22

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instance.cpp`  (+0/-217)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp`  (+114/-10)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_i8_instance.hpp`  (+118/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_instance.hpp`  (+0/-102)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_irregular_instance.cpp`  (+96/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_instance.cpp`  (+95/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_interwave_irregular_instance.cpp`  (+95/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_interwave_instance.cpp`  (+81/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v2_instance.cpp`  (+79/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v1_instance.hpp`  (+59/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v1_interwave_instance.hpp`  (+59/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v2_instance.hpp`  (+59/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v2_irregular_instance.cpp`  (+46/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`  (+26/-4)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v1_interwave_default_instance.cpp`  (+27/-0)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/gpu/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v1_instance.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/include/ck/library/tensor_operation_instance/gpu/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v1_interwave_instance.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/include/ck/library/tensor_operation_instance/gpu/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v2_instance.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`**
```
void add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_default_instances(
void add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_interwave_default_instances(
std::vector<std::unique_ptr<
DeviceGemm<Row, Row, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances);
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp`**
```
void add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_instances(
void add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_irregular_instances(
std::vector<std::unique_ptr<
DeviceGemmSplitK<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
```
