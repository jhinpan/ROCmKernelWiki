# Diff summary

- **files changed:** 15
- **lines:** +1215 / -114
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`  (+751/-38)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`  (+163/-60)
- `example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp`  (+200/-8)
- `include/ck_tile/core/arch/workgroup_barrier.hpp`  (+65/-0)
- `include/ck_tile/core/arch/arch.hpp`  (+9/-0)
- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_problem.hpp`  (+6/-3)
- `example/ck_tile/13_moe_sorting/script/smoke_test.sh`  (+6/-0)
- `example/ck_tile/15_fused_moe/instances/fused_moe_api.cpp`  (+6/-0)
- `example/ck_tile/13_moe_sorting/moe_sorting.cpp`  (+1/-2)
- `example/ck_tile/15_fused_moe/main.cpp`  (+2/-1)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.hpp`  (+1/-1)
- `example/ck_tile/15_fused_moe/fused_moe.hpp`  (+2/-0)
- `include/ck_tile/core/config.hpp`  (+1/-1)
- `example/ck_tile/15_fused_moe/fused_moesorting.hpp`  (+1/-0)
- `include/ck_tile/core.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/13_moe_sorting/moe_sorting.cpp`**
```
ck_tile::index_t workspace_size = moe_sorting_get_workspace_size(tokens, num_experts, topk);
```

**`example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`**
```
if(moe_sorting_get_workspace_size(a.tokens, a.num_experts, a.topk) != 0)
[&]() {                                                                                         \
constexpr ck_tile::index_t unroll_num = unroll_num_;                                        \
constexpr bool expert_masking         = expert_masking_;                                    \
```

**`example/ck_tile/13_moe_sorting/moe_sorting_api.hpp`**
```
int moe_sorting_get_workspace_size(int tokens, int num_experts, int topk);
```

**`example/ck_tile/15_fused_moe/fused_moe.hpp`**
```
int fused_moe_get_workspace_size(int tokens, int num_experts, int topk);
```

**`example/ck_tile/15_fused_moe/fused_moesorting.hpp`**
```
int fused_moe_get_workspace_size(int tokens, int num_experts, int topk);
```
