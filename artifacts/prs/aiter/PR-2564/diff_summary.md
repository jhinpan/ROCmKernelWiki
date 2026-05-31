# Diff summary

- **files changed:** 12
- **lines:** +25 / -27
- **kernel-ish files:** 12

## Files (by churn)

- `csrc/kernels/sample_kernels.cu`  (+6/-6)
- `csrc/py_itfs_ck/mha_batch_prefill_kernels.cu`  (+2/-2)
- `csrc/py_itfs_ck/mha_bwd_kernels.cu`  (+2/-2)
- `csrc/py_itfs_ck/mha_fwd_kernels.cu`  (+2/-2)
- `csrc/py_itfs_ck/mha_varlen_bwd_kernels.cu`  (+2/-2)
- `csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu`  (+2/-2)
- `csrc/py_itfs_cu/asm_mha_bwd.cu`  (+2/-2)
- `csrc/py_itfs_cu/asm_mha_fwd.cu`  (+2/-2)
- `csrc/py_itfs_cu/asm_mha_varlen_bwd.cu`  (+2/-2)
- `csrc/py_itfs_cu/asm_mha_varlen_fwd.cu`  (+2/-2)
- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages.cu`  (+0/-2)
- `csrc/kernels/custom_kernels.cu`  (+1/-1)

## Key added lines (kernel files)

**`csrc/kernels/custom_kernels.cu`**
```
matrixMultiplyShared<<<dimGrid, dimBlock, 0, stream>>>(
```

**`csrc/kernels/sample_kernels.cu`**
```
greedy_sample_kernel<input_dtype, block_size, warpSize, 16><<<grid, block, 0, stream>>>(
<<<grid, block, 0, stream>>>(reinterpret_cast<input_dtype*>(input.data_ptr()),
<<<grid, block, 0, stream>>>(reinterpret_cast<input_dtype*>(input.data_ptr()),
<<<grid, block, 0, stream>>>(reinterpret_cast<input_dtype*>(input.data_ptr()),
```

**`csrc/py_itfs_ck/mha_batch_prefill_kernels.cu`**
```
const hipStream_t stream = at::hip::getCurrentHIPStream();
aiter::ParsePhiloxCudaState, dim3(1), dim3(64), 0, stream, philox_args, rng_state_ptr);
```

**`csrc/py_itfs_ck/mha_bwd_kernels.cu`**
```
auto stream = at::hip::getCurrentHIPStream();
aiter::ParsePhiloxCudaState, dim3(1), dim3(64), 0, stream,
```

**`csrc/py_itfs_ck/mha_fwd_kernels.cu`**
```
const hipStream_t stream = at::hip::getCurrentHIPStream();
aiter::ParsePhiloxCudaState, dim3(1), dim3(64), 0, stream, philox_args, rng_state_ptr);
```
