# Diff summary

- **files changed:** 30
- **lines:** +300 / -194
- **kernel-ish files:** 29

## Files (by churn)

- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_fwd_api.cpp`  (+55/-45)
- `example/ck_tile/14_moe_smoothquant/script/smoke_test.sh`  (+27/-25)
- `example/ck_tile/14_moe_smoothquant/moe_smoothquant.cpp`  (+22/-11)
- `example/ck_tile/14_moe_smoothquant/moe_smoothquant.hpp`  (+10/-20)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_instance_common.hpp`  (+11/-8)
- `include/ck_tile/core/utility/unary_element_function.hpp`  (+9/-7)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n1024_instance.cpp`  (+9/-4)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n1536_instance.cpp`  (+9/-4)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n2048_instance.cpp`  (+9/-4)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n512_instance.cpp`  (+9/-4)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_fp16_n1024_instance.cpp`  (+9/-4)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_fp16_n1536_instance.cpp`  (+9/-4)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_fp16_n512_instance.cpp`  (+9/-4)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n3072_instance.cpp`  (+8/-4)
- `example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n4096_instance.cpp`  (+8/-4)

## Key added lines (kernel files)

**`example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n1024_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 1, 2,  128, 8,  true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 2, 2,  128, 4,  true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 4, 2,  128, 2,  true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 4, 1,  256, 1,  true, false>>(const S&, A);
```

**`example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n1536_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 3, 4,  64, 8, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 3, 2, 128, 4, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 3, 1, 256, 2, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 6, 1, 256, 1, true, false>>(const S&, A);
```

**`example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n2048_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 1, 1, 256, 8, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 2, 1, 256, 4, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 4, 1, 256, 2, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 8, 1, 256, 1, true, false>>(const S&, A);
```

**`example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n256_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1,  1,  4, 64, 4, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1,  2,  4, 64, 2, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1,  4,  4, 64, 1, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::fp8_t, 1,  1,  4, 64, 4, true, false>>(const S&, A);
```

**`example/ck_tile/14_moe_smoothquant/instances/moe_smoothquant_bf16_n3072_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 3, 1,  128, 8, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 3, 1,  256, 4, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 6, 1,  256, 2, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 3, 1, 1024, 1, true, false>>(const S&, A);
```
