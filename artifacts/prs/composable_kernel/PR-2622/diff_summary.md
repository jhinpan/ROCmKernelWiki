# Diff summary

- **files changed:** 24
- **lines:** +1227 / -1102
- **kernel-ish files:** 22

## Files (by churn)

- `test/ck_tile/permute/permute_utils.inc`  (+0/-490)
- `test/ck_tile/permute/test_permute_util.hpp`  (+328/-0)
- `test/ck_tile/moe_smoothquant/moe_smoothquant.inc`  (+0/-317)
- `test/ck_tile/permute/test_permute_cases.inc`  (+279/-0)
- `test/ck_tile/moe_smoothquant/test_moe_smoothquant_util.hpp`  (+218/-0)
- `test/ck_tile/moe_smoothquant/test_moe_smoothquant_cases.inc`  (+206/-0)
- `test/ck_tile/permute/alternative_impl/matrix_core_swizzle.hpp`  (+114/-3)
- `test/ck_tile/permute/alternative_impl/matrix_core_swizzle.cpp`  (+0/-101)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_fwd_api.cpp`  (+27/-25)
- `test/ck_tile/permute/permute_fp16.cpp`  (+0/-29)
- `test/ck_tile/permute/permute_fp32.cpp`  (+0/-29)
- `test/ck_tile/permute/permute_fp8.cpp`  (+0/-29)
- `test/ck_tile/permute/CMakeLists.txt`  (+2/-14)
- `test/ck_tile/moe_smoothquant/test_moe_smoothquant.cpp`  (+14/-0)
- `test/ck_tile/permute/test_permute.cpp`  (+14/-0)

## Key added lines (kernel files)

**`test/ck_tile/moe_smoothquant/instances/moe_smoothquant_fwd_api.cpp`**
```
float moe_smoothquant_dispatch(moe_smoothquant_args a, const ck_tile::stream_config& s)
template <>
float moe_smoothquant<ck_tile::fp16_t, ck_tile::int8_t>(moe_smoothquant_args a,
const ck_tile::stream_config& s)
```

**`test/ck_tile/moe_smoothquant/moe_smoothquant.hpp`**
```
template <typename InputType, typename OutputType>
float moe_smoothquant(moe_smoothquant_args, const ck_tile::stream_config&);
```

**`test/ck_tile/moe_smoothquant/test_moe_smoothquant.cpp`**
```
TYPED_TEST_SUITE(TestCkTileMoeSmoothquant, KernelTypesMoeSmoothquant);
```

**`test/ck_tile/moe_smoothquant/test_moe_smoothquant_cases.inc`**
```
TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t99_h13)
ck_tile::index_t tokens      = 99;
ck_tile::index_t hidden_size = 13;
this->Run(tokens, hidden_size);
```

**`test/ck_tile/moe_smoothquant/test_moe_smoothquant_types.hpp`**
```
using KernelTypesMoeSmoothquant = ::testing::Types<std::tuple<ck_tile::bf16_t, ck_tile::fp8_t>,
std::tuple<ck_tile::bf16_t, ck_tile::int8_t>,
std::tuple<ck_tile::fp16_t, ck_tile::fp8_t>,
std::tuple<ck_tile::fp16_t, ck_tile::int8_t>>;
```
