# Diff summary

- **files changed:** 11
- **lines:** +495 / -162
- **kernel-ish files:** 9

## Files (by churn)

- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`  (+152/-24)
- `example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp`  (+109/-54)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`  (+104/-54)
- `example/ck_tile/13_moe_sorting/moe_sorting.cpp`  (+46/-13)
- `example/ck_tile/15_fused_moe/main.cpp`  (+45/-6)
- `example/ck_tile/13_moe_sorting/README.md`  (+19/-9)
- `example/ck_tile/13_moe_sorting/script/smoke_test.sh`  (+11/-1)
- `include/ck_tile/host/reference/reference_moe_sorting.hpp`  (+3/-1)
- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_problem.hpp`  (+4/-0)
- `example/ck_tile/15_fused_moe/fused_moe.hpp`  (+1/-0)
- `example/ck_tile/15_fused_moe/instances/fused_moe_api.cpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/13_moe_sorting/moe_sorting.cpp`**
```
arg_parser.insert("v", "1", "turn CPU validation on (1) or off (0).")
.insert("pr_i", "int32", "index data type.  Only int32 is currently supported.")
.insert("pr_w", "fp32", "output weight data type. Only fp32 is currently supported.")
.insert("t",
```

**`example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`**
```
sub_token_tile_, sub_token_onshot_, local_expert_masking_, local_token_)                            \
constexpr bool local_token                = local_token_;                                           \
local_expert_masking,      \
local_token>;              \
```

**`example/ck_tile/15_fused_moe/fused_moe.hpp`**
```
const void* local_tokens;          // [1] if not nullptr, tokens read from here
```

**`example/ck_tile/15_fused_moe/instances/fused_moe_api.cpp`**
```
a.local_tokens,
```

**`example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp`**
```
sub_token_tile_, sub_token_onshot_, local_expert_masking_, local_token_)                            \
constexpr bool local_token                = local_token_;                                           \
local_expert_masking,      \
local_token>;              \
```
