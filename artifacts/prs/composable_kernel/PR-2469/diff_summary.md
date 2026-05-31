# Diff summary

- **files changed:** 9
- **lines:** +578 / -93
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`  (+354/-37)
- `example/ck_tile/13_moe_sorting/moe_sorting.cpp`  (+99/-30)
- `example/ck_tile/13_moe_sorting/script/smoke_test.sh`  (+45/-0)
- `example/ck_tile/15_fused_moe/instances/fused_moe_api.cpp`  (+24/-18)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`  (+36/-3)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.hpp`  (+9/-3)
- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_problem.hpp`  (+8/-0)
- `example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp`  (+2/-1)
- `example/ck_tile/15_fused_moe/main.cpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/13_moe_sorting/moe_sorting.cpp`**
```
.insert("moe_buf_interm_dim", "0", "interm_dim(col) of the following fmoe buf")
"moe_buf_elem_bytes", "2", "fmoe buf element byte size, 1:8bit, 2:16bit, 4:32bit...")
.insert("ci",
"clear workspace inside API or not(if \"0\", require manually clear outside)")
```

**`example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`**
```
if(moe_sorting_get_workspace_size(a.tokens, a.num_experts, a.topk, t.dispatch_policy) != 0)
maybe_clear_workspace,                                      \
maybe_clear_workspace,                                      \
maybe_clear_workspace,                                      \
```

**`example/ck_tile/13_moe_sorting/moe_sorting_api.hpp`**
```
std::string weight_type;         // currently always float
bool local_expert_masking;       // if mask experts as local expert
bool clear_workspace_inside_api; // if true, no need clear workspace outsize (will take care of
int dispatch_policy; // 0 - let the API choose kernel for you. 1 - always use single kerenl. 2 -
```

**`example/ck_tile/15_fused_moe/instances/fused_moe_api.cpp`**
```
return ck_tile::moe_sorting_get_workspace_size(
tokens, num_experts, topk, 0 /*dispatch policy*/);
auto a0 = fused_moesorting_args
a.topk_ids_ptr,              // const void* p_topk_ids;
```

**`example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp`**
```
return ck_tile::moe_sorting_get_workspace_size(
tokens, num_experts, topk, 0 /*dispatch policy*/);
```
