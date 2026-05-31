# Diff summary

- **files changed:** 16
- **lines:** +2026 / -865
- **kernel-ish files:** 14

## Files (by churn)

- `test/ck_tile/moe_sorting/test_moe_sorting_cases.inc`  (+1211/-0)
- `test/ck_tile/moe_sorting/moe_sorting_fp32.cpp`  (+0/-544)
- `test/ck_tile/moe_sorting/test_moe_sorting_util.hpp`  (+356/-0)
- `test/ck_tile/smoothquant/smoothquant.inc`  (+0/-273)
- `test/ck_tile/smoothquant/test_smoothquant_cases.inc`  (+206/-0)
- `test/ck_tile/smoothquant/test_smoothquant_util.hpp`  (+181/-0)
- `test/ck_tile/smoothquant/instances/smoothquant_fwd_api.cpp`  (+10/-14)
- `test/ck_tile/moe_sorting/CMakeLists.txt`  (+13/-8)
- `test/ck_tile/moe_sorting/test_moe_sorting.cpp`  (+14/-0)
- `test/ck_tile/smoothquant/test_smoothquant.cpp`  (+14/-0)
- `test/ck_tile/smoothquant/smoothquant_bf16.cpp`  (+0/-11)
- `test/ck_tile/smoothquant/smoothquant_fp16.cpp`  (+0/-11)
- `test/ck_tile/smoothquant/test_smoothquant_types.hpp`  (+9/-0)
- `test/ck_tile/moe_sorting/test_moe_sorting_types.hpp`  (+8/-0)
- `test/ck_tile/smoothquant/CMakeLists.txt`  (+2/-3)

## Key added lines (kernel files)

**`test/ck_tile/moe_sorting/test_moe_sorting.cpp`**
```
TYPED_TEST_SUITE(TestCkTileMoeSorting, KernelTypesMoeSorting);
```

**`test/ck_tile/moe_sorting/test_moe_sorting_cases.inc`**
```
TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase1)
int tokens       = 80;
int local_tokens = -1;
int num_experts  = 17;
```

**`test/ck_tile/moe_sorting/test_moe_sorting_types.hpp`**
```
using KernelTypesMoeSorting = ::testing::Types<std::tuple<float, ck_tile::index_t>>;
```

**`test/ck_tile/moe_sorting/test_moe_sorting_util.hpp`**
```
template <typename IndexType>
void topid_unique_gen(
std::vector<IndexType>& host_tensor, int tokens, int topk, int num_expert, int seed)
size_t total_size = topk * tokens;
```

**`test/ck_tile/smoothquant/instances/smoothquant_fwd_api.cpp`**
```
float smoothquant_dispatch(smoothquant_args a, const ck_tile::stream_config& s)
template <>
float smoothquant<ck_tile::fp16_t>(smoothquant_args a, const ck_tile::stream_config& s)
return smoothquant_dispatch<ck_tile::fp16_t>(a, s);
```
