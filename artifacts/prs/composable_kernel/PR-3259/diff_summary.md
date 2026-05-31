# Diff summary

- **files changed:** 39
- **lines:** +173 / -172
- **kernel-ish files:** 10

## Files (by churn)

- `example/ck_tile/03_gemm/CMakeLists.txt`  (+21/-19)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_large_tensor_invoker.hpp`  (+22/-18)
- `example/ck_tile/18_flatmm/CMakeLists.txt`  (+21/-16)
- `example/ck_tile/15_fused_moe/CMakeLists.txt`  (+16/-16)
- `example/ck_tile/20_grouped_convolution/CMakeLists.txt`  (+16/-14)
- `example/ck_tile/41_batched_contraction/run_batched_contraction_example.inc`  (+14/-14)
- `example/ck_tile/17_grouped_gemm/CMakeLists.txt`  (+13/-11)
- `example/ck_tile/35_batched_transpose/CMakeLists.txt`  (+9/-8)
- `example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp`  (+0/-16)
- `tutorial/ck_tile/01_naive_gemm/host_level/practice_gemm_host_pipeline_agmem_bgmem_creg.hpp`  (+4/-4)
- `include/ck_tile/ops/pooling/kernel/pool_kernel.hpp`  (+4/-2)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+2/-2)
- `example/ck_tile/05_reduce/CMakeLists.txt`  (+2/-2)
- `example/ck_tile/10_rmsnorm2d/CMakeLists.txt`  (+2/-2)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/CMakeLists.txt`  (+2/-2)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`**
```
ck_tile::HostTensor<BDataType> b_shuffle_host = ck_tile::shuffle_b<GemmConfig>(b_k_n);
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
return ck_tile::shuffle_b_permuteN<GemmConfig>(b_k_n);
return ck_tile::shuffle_b<GemmConfig>(b_k_n);
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`**
```
ck_tile::HostTensor<BDataType> b_shuffle_host =
ck_tile::shuffle_b<GemmConfig>(b_k_n_tensors[i]);
```

**`example/ck_tile/18_flatmm/run_grouped_flatmm_example.inc`**
```
ck_tile::shuffle_b<FlatmmConfig, BDataType>(b_k_n_tensor);
ck_tile::shuffle_b<FlatmmConfig, BDataType>(b_k_n_tensor);
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_forward_large_tensor_invoker.hpp`**
```
const auto Run = [&](const auto has_hot_loop_,
const auto tail_number_,
const auto memory_operation_,
const auto enable_split_image_) {
```
