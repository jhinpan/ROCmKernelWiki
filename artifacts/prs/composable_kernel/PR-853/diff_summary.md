# Diff summary

- **files changed:** 20
- **lines:** +956 / -42
- **kernel-ish files:** 18

## Files (by churn)

- `client_example/20_splitk_gemm/splitK_gemm_fp16_f8.cpp`  (+225/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp`  (+88/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instance.cpp`  (+71/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instance.cpp`  (+71/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_instance.cpp`  (+67/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f8_f16_f16_mk_nk_mn_instance.cpp`  (+67/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f8_f16_km_kn_mn_instance.cpp`  (+62/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f8_f16_km_nk_mn_instance.cpp`  (+62/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f8_f16_f16_km_kn_mn_instance.cpp`  (+62/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f8_f16_f16_km_nk_mn_instance.cpp`  (+62/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/CMakeLists.txt`  (+28/-10)
- `profiler/src/profile_gemm_splitk.cpp`  (+36/-1)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`  (+18/-16)
- `profiler/include/profiler/profile_gemm_splitk_impl.hpp`  (+14/-1)
- `example/35_splitK_gemm/splitK_gemm_xdl_bfp16.cpp`  (+6/-5)

## Key added lines (kernel files)

**`client_example/20_splitk_gemm/splitK_gemm_fp16_f8.cpp`**
```
using F8  = ck::f8_t;
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
```

**`example/35_splitK_gemm/splitK_gemm_xdl_bfp16.cpp`**
```
using ComputeType = BF16;
< ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,    Gem
```

**`example/35_splitK_gemm/splitK_gemm_xdl_int8.cpp`**
```
using ComputeType = int8_t;
< ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,    Gem
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`**
```
index_t CBlockTransferScalarPerVector_NWaveNPerXDL,
typename ComputeType = CDataType>
ADataType,
BDataType,
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`**
```
ADataType,
BDataType,
```
