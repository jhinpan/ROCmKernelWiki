# Diff summary

- **files changed:** 36
- **lines:** +1321 / -11
- **kernel-ish files:** 30

## Files (by churn)

- `example/ck_tile/14_moe_smoothquant/moe_smoothquant.cpp`  (+264/-0)
- `include/ck_tile/ops/smoothquant/kernel/moe_smoothquant_kernel.hpp`  (+205/-0)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_fwd_api.cpp`  (+145/-0)
- `example/ck_tile/14_moe_smoothquant/moe_smoothquant.hpp`  (+114/-0)
- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`  (+77/-6)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_instance_common.hpp`  (+62/-0)
- `example/ck_tile/14_moe_smoothquant/script/perf_test.sh`  (+37/-0)
- `include/ck_tile/core/numeric/bfloat16.hpp`  (+36/-0)
- `example/ck_tile/14_moe_smoothquant/script/smoke_test.sh`  (+30/-0)
- `include/ck_tile/host/reference/reference_moe_sorting.hpp`  (+24/-5)
- `example/ck_tile/14_moe_smoothquant/CMakeLists.txt`  (+25/-0)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n1024_instance.cpp`  (+22/-0)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_fp16_n1024_instance.cpp`  (+22/-0)
- `example/ck_tile/14_moe_smoothquant/README.md`  (+15/-0)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n2048_instance.cpp`  (+14/-0)

## Key added lines (kernel files)

**`example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n1024_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1,  2,  4,  64, 8,  true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1,  4,  4,  64, 4,  true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1,  8,  4,  64, 2,  true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1, 16,  4,  64, 1,  true, false>>(const S&, A);
```

**`example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n1536_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1, 3, 4,  64, 8, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1, 3, 2, 128, 4, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1, 3, 1, 256, 2, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1, 6, 1, 256, 1, true, false>>(const S&, A);
```

**`example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n2048_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t,  1, 1, 1, 256, 8, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t,  1, 2, 1, 256, 4, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t,  1, 4, 1, 256, 2, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t,  1, 8, 1, 256, 1, true, false>>(const S&, A);
```

**`example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n256_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1,  1,  4, 64, 4, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1,  2,  4, 64, 2, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1,  4,  4, 64, 1, true, false>>(const S&, A);
```

**`example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n3072_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t,  1, 3, 1,  128, 8, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t,  1, 3, 1,  256, 4, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t,  1, 6, 1,  256, 2, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t,  1, 3, 1, 1024, 1, true, false>>(const S&, A);
```
