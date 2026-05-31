# Diff summary

- **files changed:** 8
- **lines:** +1043 / -31
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`  (+819/-5)
- `example/ck_tile/13_moe_sorting/moe_sorting.cpp`  (+94/-18)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`  (+96/-8)
- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_problem.hpp`  (+17/-0)
- `example/ck_tile/15_fused_moe/main.cpp`  (+7/-0)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.hpp`  (+6/-0)
- `example/ck_tile/15_fused_moe/fused_moe.hpp`  (+3/-0)
- `example/ck_tile/15_fused_moe/instances/fused_moe_api.cpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/13_moe_sorting/moe_sorting.cpp`**
```
ck_tile::index_t workspace_size = moe_sorting_get_workspace_size(tokens, num_experts);
ck_tile::DeviceMem moe_sorting_ws(workspace_size != 0 ? workspace_size : 0);
if(workspace_size != 0)
moe_sorting_ws.SetZero(); // note, clear here!!!!
```

**`example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`**
```
if(moe_sorting_get_workspace_size(a.tokens, a.num_experts) != 0)
return moe_sorting_mp(t, a, s);
using index_t                = ck_tile::index_t;
using ms_weight_type         = float;
```

**`example/ck_tile/13_moe_sorting/moe_sorting_api.hpp`**
```
int moe_sorting_get_workspace_size(int tokens, int num_experts);
float moe_sorting_mp(moe_sorting_trait t, moe_sorting_args a, ck_tile::stream_config s);
```

**`example/ck_tile/15_fused_moe/fused_moe.hpp`**
```
void* ws_ptr;                      // size is moe_sorting_get_workspace_size()
```

**`example/ck_tile/15_fused_moe/instances/fused_moe_api.cpp`**
```
a.ws_ptr,                                    // void* p_ws;
```
