# Diff summary

- **files changed:** 19
- **lines:** +89 / -24
- **kernel-ish files:** 19

## Files (by churn)

- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+15/-8)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp`  (+12/-9)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp`  (+13/-6)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn.hpp`  (+6/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn.hpp`  (+6/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn.hpp`  (+6/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn.hpp`  (+5/-1)
- `example/01_gemm/gemm_xdl_streamk.cpp`  (+4/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn.hpp`  (+2/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f16_f8_f16/device_gemm_xdl_universal_f16_f8_f16_mk_kn_mn.hpp`  (+2/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f16_f8_f16/device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn.hpp`  (+2/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn.hpp`  (+2/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn.hpp`  (+2/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_batched/device_batched_gemm_xdl_universal_bf16_bf16_bf16/device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn.hpp`  (+2/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_reduce/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_streamk.cpp`**
```
< ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,    256
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
int n_warmup                 = arg_parser.get_int("warmup");
int n_repeat                 = arg_parser.get_int("repeat");
ck_tile::index_t init_method = arg_parser.get_int("init");
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp`**
```
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,DsLayout,ELayout,int8_t,int8_t,int32_t,   in
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp`**
```
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,   
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,   
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,   
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,   
```

**`library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_nk_mn.hpp`**
```
DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,   F16,  F16,   F32,     F16,      PassThrough, PassThrough, 
```
