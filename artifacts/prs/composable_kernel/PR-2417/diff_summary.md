# Diff summary

- **files changed:** 14
- **lines:** +2195 / -0
- **kernel-ish files:** 11

## Files (by churn)

- `test/ck_tile/moe_sorting/moe_sorting_fp32.cpp`  (+538/-0)
- `test/ck_tile/permute/permute_utils.inc`  (+490/-0)
- `test/ck_tile/moe_sorting/moe_sorting_api.cpp`  (+444/-0)
- `test/ck_tile/permute/alternative_impl/matrix_core_swizzle_kernel.hpp`  (+413/-0)
- `test/ck_tile/permute/alternative_impl/matrix_core_swizzle.cpp`  (+101/-0)
- `test/ck_tile/moe_sorting/moe_sorting_api.hpp`  (+33/-0)
- `test/ck_tile/permute/CMakeLists.txt`  (+33/-0)
- `test/ck_tile/permute/permute_fp16.cpp`  (+29/-0)
- `test/ck_tile/permute/permute_fp32.cpp`  (+29/-0)
- `test/ck_tile/permute/permute_fp8.cpp`  (+29/-0)
- `test/ck_tile/permute/alternative_impl/matrix_core_swizzle.hpp`  (+20/-0)
- `test/ck_tile/permute/permute.hpp`  (+19/-0)
- `test/ck_tile/moe_sorting/CMakeLists.txt`  (+15/-0)
- `test/ck_tile/CMakeLists.txt`  (+2/-0)

## Key added lines (kernel files)

**`test/ck_tile/moe_sorting/moe_sorting_api.cpp`**
```
constexpr ck_tile::index_t unroll_num  = unroll_num_;                             \
constexpr ck_tile::index_t expert_tile = expert_tile_;                            \
using ms_problem =                                                                \
ck_tile::MoeSortingProblem<index_t, ms_weight_type, unroll_num, expert_tile>; \
```

**`test/ck_tile/moe_sorting/moe_sorting_api.hpp`**
```
struct moe_sorting_trait
std::string index_type;
std::string weight_type;         // currently always float
bool local_expert_masking;       // if mask experts as local expert
```

**`test/ck_tile/moe_sorting/moe_sorting_fp32.cpp`**
```
auto create_args(int argc, char* argv[], int index = 0)
ck_tile::ArgParser arg_parser;
arg_parser.insert("v", "1", "turn CPU validation on (1) or off (0).")
.insert("pr_i", "int32", "index data type.  Only int32 is currently supported.")
```

**`test/ck_tile/permute/alternative_impl/matrix_core_swizzle.cpp`**
```
float matrix_core_swizzle(matrix_core_swizzle_traits t,
matrix_core_swizzle_args a,
const ck_tile::stream_config& s)
if(t.data_type.compare("fp16") == 0)
```

**`test/ck_tile/permute/alternative_impl/matrix_core_swizzle.hpp`**
```
struct matrix_core_swizzle_traits
std::string data_type; // fp16 only
std::string inst;      // 32x32x8, 16x16x16
std::string permute;   //
```
