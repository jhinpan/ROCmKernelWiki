# Diff summary

- **files changed:** 34
- **lines:** +1956 / -324
- **kernel-ish files:** 32

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp`  (+356/-188)
- `profiler/include/profiler/profile_grouped_gemm_multiply_tile_loop_impl.hpp`  (+347/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_instance.cpp`  (+129/-61)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop_multiply.hpp`  (+167/-6)
- `profiler/src/profile_grouped_gemm_multiply_tile_loop.cpp`  (+133/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d.hpp`  (+83/-29)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn.hpp`  (+93/-0)
- `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp`  (+45/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_multiply_bias_fastgelu_bf16_i8_bf16_mk_kn_mn_instance.cpp`  (+41/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_multiply_bias_bf16_i8_bf16_mk_kn_mn_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_multiply_fastgelu_bf16_i8_bf16_mk_kn_mn_instance.cpp`  (+39/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_default_instance.cpp`  (+36/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_kpadding_instance.cpp`  (+36/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_mnkpadding_instance.cpp`  (+36/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_mnpadding_instance.cpp`  (+36/-0)

## Key added lines (kernel files)

**`example/15_grouped_gemm/grouped_gemm_multiple_d_xdl_fp16.cpp`**
```
< ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AEle
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1.hpp`**
```
__host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
__host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
__host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
__host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp`**
```
__host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
__host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
__host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
__host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp`**
```
__host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
__host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v4.hpp`**
```
__host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
__host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
```
