# Diff summary

- **files changed:** 27 (diff was byte-capped; summary is partial)
- **lines:** +1616 / -0
- **kernel-ish files:** 25

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_f16/device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn.hpp`  (+306/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_nk_mn.hpp`  (+279/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_preshuffle.hpp`  (+151/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_preshuffle.inc`  (+122/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/CMakeLists.txt`  (+82/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instance_p1.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instance_p2.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instance_p3.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instance_p4.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instance_p5.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_nk_mn_comp_default_instance_p1.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_nk_mn_comp_default_instance_p2.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instance_p6.cpp`  (+32/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p1_default_instance.cpp`  (+32/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p2_default_instance.cpp`  (+32/-0)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_preshuffle.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_preshuffle.inc`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instance_p1.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instance_p2.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instance_p3.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
