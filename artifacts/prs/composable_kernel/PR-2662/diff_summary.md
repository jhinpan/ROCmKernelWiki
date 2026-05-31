# Diff summary

- **files changed:** 8
- **lines:** +176 / -24
- **kernel-ish files:** 8

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn.hpp`  (+84/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp`  (+50/-22)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_default_instance.cpp`  (+7/-1)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_kpadding_instance.cpp`  (+7/-1)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v1_default_instance.cpp`  (+7/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v1_kpadding_instance.cpp`  (+7/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v2_default_instance.cpp`  (+7/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v2_kpadding_instance.cpp`  (+7/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp`**
```
enum struct Arch : bool
is_gfx950_build = true,
is_gfx950_build = false,
if constexpr(((GridwiseGemm::AK1Number >= 32 || GridwiseGemm::BK1Number >= 32) &&
```

**`library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn.hpp`**
```
template <GemmSpecialization GemmSpec>
using device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_instances_dr = std::tuple<
DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F8,     F8,    BF16,   F32,     BF16,  PassThrough, PassThrough, 
DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F8,     F8,    BF16,   F32,     BF16,  PassThrough, PassThrough, 
```

**`library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_default_instance.cpp`**
```
if(ck::get_device_name() == "gfx950")
add_device_operation_instances(
instances,
device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_instances_dr<GemmDefault>{});
```

**`library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_kpadding_instance.cpp`**
```
if(ck::get_device_name() == "gfx950")
add_device_operation_instances(
instances,
device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_comp_instances_dr<GemmKPadding>{});
```

**`library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v1_default_instance.cpp`**
```
if(ck::get_device_name() == "gfx950")
add_device_operation_instances(
instances,
device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_instances_dr<Intrawave,
```
