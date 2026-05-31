# Diff summary

- **files changed:** 59
- **lines:** +2916 / -1041
- **kernel-ish files:** 53

## Files (by churn)

- `include/ck_tile/ops/layernorm2d/kernel/layernorm2d_fwd_kernel.hpp`  (+148/-351)
- `include/ck_tile/ops/welford/block/block_welford.hpp`  (+362/-0)
- `include/ck_tile/core/tensor/sweep_tile.hpp`  (+278/-0)
- `example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`  (+117/-119)
- `example/66_complex_contraction_bilinear/run_complex_contraction_bilinear_example.inc`  (+110/-113)
- `include/ck_tile/core/utility/functional_with_tuple.hpp`  (+173/-0)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_two_pass.hpp`  (+160/-0)
- `include/ck_tile/core/tensor/tile_distribution.hpp`  (+35/-123)
- `example/ck_tile/02_layernorm2d/instances/layernorm2d_fwd_api.cpp`  (+155/-0)
- `include/ck_tile/ops/welford/warp/warp_welford.hpp`  (+0/-154)
- `include/ck_tile/core/container/sequence.hpp`  (+122/-0)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_one_pass.hpp`  (+119/-0)
- `example/ck_tile/05_reduce/reduce.hpp`  (+118/-0)
- `example/ck_tile/02_layernorm2d/layernorm2d_fwd.hpp`  (+104/-13)
- `include/ck_tile/ops/welford/thread/thread_welford.hpp`  (+24/-89)

## Key added lines (kernel files)

**`example/66_complex_contraction_bilinear/run_complex_contraction_bilinear_example.inc`**
```
case 0: break;
a_ms_ks_re.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
b_ns_ks_re.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
d_ms_ns_re.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
```

**`example/ck_tile/02_layernorm2d/instances/layernorm2d_fwd_api.cpp`**
```
template <typename DataType_,
ck_tile::index_t Repeat_M_,         // each thread repeat along M
ck_tile::index_t Repeat_N_,         // each thread repeat along N
ck_tile::index_t ThreadPerBlock_M_, // num threads along M
```

**`example/ck_tile/02_layernorm2d/instances/layernorm2d_fwd_bf16_n1024_instance.cpp`**
```
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t, 1,  2,  4,  64, 8,  true , false, false>>(const S&, A);
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t, 1,  4,  4,  64, 4,  true , false, false>>(const S&, A);
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t, 1,  8,  4,  64, 2,  true , false, false>>(const S&, A);
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t, 1, 16,  4,  64, 1,  true , false, false>>(const S&, A);
```

**`example/ck_tile/02_layernorm2d/instances/layernorm2d_fwd_bf16_n1536_instance.cpp`**
```
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t,  1, 3, 4,   64, 8,  true,  false, false>>(const S&, A);
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t,  1, 3, 2,  128, 4,  true,  false, false>>(const S&, A);
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t,  1, 3, 1,  256, 2,  true,  false, false>>(const S&, A);
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t,  1, 6, 1,  256, 1,  true,  false, false>>(const S&, A);
```

**`example/ck_tile/02_layernorm2d/instances/layernorm2d_fwd_bf16_n2048_instance.cpp`**
```
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t,  1, 1, 1,  256, 8,  true,  false, false>>(const S&, A);
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t,  1, 2, 1,  256, 4,  true,  false, false>>(const S&, A);
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t,  1, 4, 1,  256, 2,  true,  false, false>>(const S&, A);
template float layernorm2d_fwd_<trait_<ck_tile::bf16_t,  1, 8, 1,  256, 1,  true,  false, false>>(const S&, A);
```
