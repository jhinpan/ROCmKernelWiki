# Diff summary

- **files changed:** 55
- **lines:** +2509 / -384
- **kernel-ish files:** 50

## Files (by churn)

- `include/ck/utility/amd_ck_fp8.hpp`  (+988/-0)
- `include/ck/utility/data_type.hpp`  (+360/-83)
- `test/data_type/test_bf8_ocp.cpp`  (+268/-0)
- `test/data_type/test_fp8_ocp.cpp`  (+250/-0)
- `include/ck/utility/type_convert.hpp`  (+143/-61)
- `test/data_type/test_custom_type.cpp`  (+158/-0)
- `test/data_type/test_fp8_fnuz.cpp`  (+83/-66)
- `test/data_type/test_bf8_fnuz.cpp`  (+73/-62)
- `test/data_type/CMakeLists.txt`  (+31/-6)
- `include/ck/library/utility/host_tensor_generator.hpp`  (+25/-6)
- `include/ck/utility/amd_buffer_addressing.hpp`  (+8/-6)
- `include/ck/utility/random_gen.hpp`  (+8/-5)
- `CMakeLists.txt`  (+10/-1)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_gemm.hpp`  (+5/-5)
- `client_example/CMakeLists.txt`  (+8/-0)

## Key added lines (kernel files)

**`example/01_gemm/common.hpp`**
```
int do_verification = 1;
```

**`example/01_gemm/run_gemm_example.inc`**
```
ck::utils::FillConstant<ADataType>{ck::type_convert<ADataType>(1.f)}(a_m_k);
ck::utils::FillConstant<BDataType>{ck::type_convert<BDataType>(1.f)}(b_k_n);
```

**`example/15_grouped_gemm/grouped_gemm_multiple_d_splitk_xdl_fp16.cpp`**
```
d_tensors[i][j].GenerateTensorValue(GeneratorTensor_3<DDataType>{0.0, 1.0});
a_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<ADataType, 0>{});
b_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<BDataType, 1>{});
d_tensors[i][j].GenerateTensorValue(GeneratorTensor_Sequential<DDataType, 0>{});
```

**`example/15_grouped_gemm/grouped_gemm_multiple_d_xdl_fp16.cpp`**
```
d_tensors[i][j].GenerateTensorValue(GeneratorTensor_3<DDataType>{0.0, 1.0});
a_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<ADataType, 0>{});
b_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<BDataType, 1>{});
d_tensors[i][j].GenerateTensorValue(GeneratorTensor_Sequential<DDataType, 0>{});
```

**`example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_bias_fp16.cpp`**
```
a_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<ADataType, 0>{});
b_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<BDataType, 1>{});
d0_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<D0DataType, 1>{});
```
