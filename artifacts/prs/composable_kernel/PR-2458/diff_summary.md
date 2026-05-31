# Diff summary

- **files changed:** 32
- **lines:** +1217 / -9
- **kernel-ish files:** 30

## Files (by churn)

- `test/ck_tile/moe_smoothquant/moe_smoothquant.inc`  (+317/-0)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_fwd_api.cpp`  (+155/-0)
- `test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_ut_cases.inc`  (+117/-2)
- `test/ck_tile/moe_smoothquant/moe_smoothquant.hpp`  (+104/-0)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_instance_common.hpp`  (+65/-0)
- `test/ck_tile/moe_smoothquant/CMakeLists.txt`  (+32/-0)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n1024_instance.cpp`  (+27/-0)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_fp16_n1024_instance.cpp`  (+27/-0)
- `test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_kernel_types.hpp`  (+13/-7)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n2048_instance.cpp`  (+19/-0)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n1536_instance.cpp`  (+18/-0)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n3072_instance.cpp`  (+18/-0)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n4096_instance.cpp`  (+18/-0)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n4096_tp_instance.cpp`  (+18/-0)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n512_instance.cpp`  (+18/-0)

## Key added lines (kernel files)

**`test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_kernel_types.hpp`**
```
using F16  = ck_tile::half_t;
using F32  = float;
using F8   = ck_tile::fp8_t;
using BF16 = ck_tile::bf16_t;
```

**`test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_ut_cases.inc`**
```
TYPED_TEST(TEST_SUITE_NAME, GemmPreshuffle_128x128x128)
if constexpr(std::is_same_v<TypeParam, F8Types>)
GTEST_SKIP() << "Skipping this test due to failures with F8";
constexpr int M           = 128;
```

**`test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n1024_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1,  2,  4,  64, 8,  true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1,  4,  4,  64, 4,  true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1,  8,  4,  64, 2,  true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1, 16,  4,  64, 1,  true, false>>(const S&, A);
```

**`test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n1536_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 3, 4,  64, 8, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 3, 2, 128, 4, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 3, 1, 256, 2, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 6, 1, 256, 1, true, false>>(const S&, A);
```

**`test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n2048_instance.cpp`**
```
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 1, 1, 256, 8, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 2, 1, 256, 4, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 4, 1, 256, 2, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1, 8, 1, 256, 1, true, false>>(const S&, A);
```
