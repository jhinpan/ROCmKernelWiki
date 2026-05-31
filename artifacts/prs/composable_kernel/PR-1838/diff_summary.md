# Diff summary

- **files changed:** 28
- **lines:** +1387 / -333
- **kernel-ish files:** 24

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v3.hpp`  (+860/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v2.hpp`  (+107/-119)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply_weight_preshuffle.hpp`  (+81/-50)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_weight_preshuffle/device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16/device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn.hpp`  (+58/-32)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_weight_preshuffle/device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16/device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn.hpp`  (+58/-32)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_bpreshuffle.cpp`  (+42/-22)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle.hpp`  (+11/-33)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_preshuffle.hpp`  (+29/-4)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_weight_preshuffle/device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16/device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_compute_default_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_weight_preshuffle/device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16/device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_compute_default_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_weight_preshuffle/CMakeLists.txt`  (+16/-12)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_selector.hpp`  (+25/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v1.hpp`  (+7/-7)
- `CMakeLists.txt`  (+0/-7)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_weight_preshuffle/device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16/device_gemm_multiply_multiply_weight_preshuffle_xdl_f8_f8_bf16_mk_mfma_mn_p1_default_instance_v2.cpp`  (+3/-2)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_bpreshuffle.cpp`**
```
<   Row, Col, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
AElementOp,  BElementOp, CDEElementOp, GemmSpec, 256,
256,   256,    128,
16,   16,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_selector.hpp`**
```
else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
static_assert(MRepeat >= 4, "MRepeat should at least be 4 in BlockGemmPipelineVersion::v3");
return BlockwiseGemmXdlops_pipeline_bpreshuffle_v3<BlkGemmPipeSche,
BlockSize,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v1.hpp`**
```
static constexpr index_t GlobalBufferNum = 2;
a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
__builtin_amdgcn_sched_barrier(0);
a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v2.hpp`**
```
static constexpr index_t PrefetchStages  = 3;
static constexpr index_t PrefillStages   = 2;
static constexpr index_t GlobalBufferNum = 2;
static_for<0, num_buffer_load_inst_b / 2, 1>{}([&](auto i) {
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v3.hpp`**
```
namespace ck {
template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
index_t BlockSize,
typename ADataType,
```
