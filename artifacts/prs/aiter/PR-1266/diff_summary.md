# Diff summary

- **files changed:** 9
- **lines:** +606 / -48
- **kernel-ish files:** 9

## Files (by churn)

- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4_bns.cuh`  (+367/-0)
- `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`  (+103/-5)
- `hsa/gfx942/fmoe_2stages/tune.py`  (+85/-11)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`  (+30/-4)
- `hsa/gfx950/fmoe_2stages/tune.py`  (+1/-17)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4.cuh`  (+8/-7)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`  (+9/-2)
- `op_tests/test_moe_2stage.py`  (+2/-1)
- `aiter/fused_moe.py`  (+1/-1)

## Key added lines (kernel files)

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`**
```
std::string moe_env_value = "0";
if (const char* env = std::getenv("AITER_MXFP4_MOE_SF")) {
moe_env_value = std::string(env);
bool use_mxfp4_moe_preshuffle = std::string(moe_env_value) == "1";
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`**
```
0: kernelInstanceGEMM1(       256,       32,          128,       128,     1,       4,        3,),
1: kernelInstanceGEMM1(       256,       64,          128,       128,     1,       4,        3,),
2: kernelInstanceGEMM1(       256,      128,          128,       128,     1,       4,        3,),
a4w4_bns_gemm1_kernels_list= {
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4.cuh`**
```
using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemmMXBPreShuffle
S<K0_A, K0_M_A, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, AK1, AK1, 1,
S<K0_B, K0_N_B, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, BK1, BK1, 1,
using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemmMXBPreShuffle
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4_bns.cuh`**
```
template <typename A0DataType,
typename B0DataType,
typename AccDataType,
typename EDataType,
```

**`csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`**
```
MoeKernel moe_stage1_heuristic_dispatch(int block_m, at::ScalarType x_dtype, at::ScalarType w_dtype, at::ScalarType y_dt
MoeKernel moe_stage2_heuristic_dispatch(int block_m, int inter_dim, at::ScalarType x_dtype, at::ScalarType w_dtype, at::
if (dtype_checker<{A0DataType}>{{}}(x_dtype)
&& dtype_checker<{B0DataType}>{{}}(w_dtype)
```
