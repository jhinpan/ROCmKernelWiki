# Diff summary

- **files changed:** 13
- **lines:** +937 / -100
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`  (+634/-59)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`  (+82/-0)
- `example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp`  (+74/-0)
- `example/ck_tile/13_moe_sorting/moe_sorting.cpp`  (+57/-6)
- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_problem.hpp`  (+52/-0)
- `include/ck_tile/ops/fused_moe/pipeline/moe_sorting_problem.hpp`  (+0/-28)
- `include/ck_tile/host/reference/reference_moe_sorting.hpp`  (+24/-2)
- `example/ck_tile/13_moe_sorting/script/smoke_test.sh`  (+8/-0)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.hpp`  (+2/-1)
- `example/ck_tile/15_fused_moe/README.md`  (+1/-1)
- `include/ck_tile/core.hpp`  (+1/-1)
- `include/ck_tile/ops/fused_moe.hpp`  (+1/-1)
- `include/ck_tile/ops/fused_moe/kernel/fused_moegemm_kernel.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/13_moe_sorting/moe_sorting.cpp`**
```
.insert("local_eid",
"a list of experts enabled as local expert. e.g. \"0,1,4,5\"\n"
"please make sure eid is in ascending order!")
bool local_expert_masking      = args.get_str("local_eid") != "-1";
```

**`example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`**
```
constexpr ck_tile::index_t sub_token_tile = sub_token_tile_;                                        \
constexpr bool sub_token_onshot           = sub_token_onshot_;                                      \
constexpr bool local_expert_masking       = local_expert_masking_;                                  \
using ms_problem                          = ck_tile::MoeSortingProblemEx<index_t,                   \
```

**`example/ck_tile/13_moe_sorting/moe_sorting_api.hpp`**
```
std::string weight_type;   // currently always float
bool local_expert_masking; // if mask experts as local expert
```

**`example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp`**
```
constexpr ck_tile::index_t sub_token_tile = sub_token_tile_;                                 \
constexpr bool sub_token_onshot           = sub_token_onshot_;                               \
using ms_problem =                                                                           \
ck_tile::MoeSortingProblemEx<index_t, ms_weight_type, sub_token_tile, sub_token_onshot>; \
```

**`include/ck_tile/host/reference/reference_moe_sorting.hpp`**
```
const HostTensor<IndexType>& local_expert_mask,
const index_t unit_size,
bool local_expert_masking,
bool skip_experts_with_zero_token = true)
```
