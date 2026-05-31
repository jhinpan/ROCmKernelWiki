# Diff summary

- **files changed:** 18
- **lines:** +2475 / -1009
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_wmma_cshuffle_v3.hpp`  (+956/-0)
- `example/29_batched_gemm_bias_e_permute/batched_gemm_bias_e_permute_wmma_fp16.cpp`  (+2/-349)
- `example/29_batched_gemm_bias_e_permute/run_batched_gemm_bias_e_permute_example.inc`  (+350/-0)
- `example/29_batched_gemm_bias_e_permute/batched_gemm_bias_e_permute_xdl_fp16.cpp`  (+2/-337)
- `profiler/include/profiler/profile_batched_contraction_multiple_d_impl.hpp`  (+309/-0)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_contraction.hpp`  (+273/-0)
- `example/25_gemm_bias_e_permute/gemm_bias_e_permute_g1m3n2k1_xdl_fp16.cpp`  (+21/-148)
- `example/25_gemm_bias_e_permute/gemm_bias_e_permute_g1m2n3k1_xdl_fp16.cpp`  (+20/-148)
- `test/batched_contraction/test_batched_contraction.cpp`  (+164/-0)
- `example/29_batched_gemm_bias_e_permute/batched_gemm_bias_e_permute_wmma_v3_fp16.cpp`  (+111/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+83/-18)
- `library/src/tensor_operation_instance/gpu/batched_gemm_bias_permute/device_batched_gemm_bias_permute_m2_n3_k1_wmma_c_shuffle_f16_f16_f16_f16_instance.cpp`  (+78/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+68/-8)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_bias_permute.hpp`  (+25/-0)
- `test/batched_contraction/CMakeLists.txt`  (+9/-0)

## Key added lines (kernel files)

**`example/25_gemm_bias_e_permute/gemm_bias_e_permute_g1m2n3k1_xdl_fp16.cpp`**
```
using ReferenceOpInstance =
ck::tensor_operation::host::ReferenceBatchedContraction_G1_M2_N3_K1<NumDimG,
ADataType,
BDataType,
```

**`example/25_gemm_bias_e_permute/gemm_bias_e_permute_g1m3n2k1_xdl_fp16.cpp`**
```
using ReferenceOpInstance =
ck::tensor_operation::host::ReferenceBatchedContraction_G1_M3_N2_K1<NumDimG,
ADataType,
BDataType,
```

**`example/29_batched_gemm_bias_e_permute/batched_gemm_bias_e_permute_wmma_fp16.cpp`**
```
int main(int argc, char* argv[]) { return !run_batched_gemm_bias_e_permute_example(argc, argv); }
```

**`example/29_batched_gemm_bias_e_permute/batched_gemm_bias_e_permute_wmma_v3_fp16.cpp`**
```
using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::make_ParallelTensorFunctor;
using ::ck::Tensor;
```

**`example/29_batched_gemm_bias_e_permute/batched_gemm_bias_e_permute_xdl_fp16.cpp`**
```
int main(int argc, char* argv[]) { return !run_batched_gemm_bias_e_permute_example(argc, argv); }
```
