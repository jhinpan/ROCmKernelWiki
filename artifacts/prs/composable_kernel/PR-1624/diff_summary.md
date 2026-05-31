# Diff summary

- **files changed:** 14
- **lines:** +770 / -0
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`  (+232/-0)
- `example/ck_tile/13_moe_sorting/moe_sorting.cpp`  (+223/-0)
- `include/ck_tile/host/reference/reference_moe_sorting.hpp`  (+78/-0)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`  (+73/-0)
- `include/ck_tile/ops/fused_moe/pipeline/moe_sorting_pipeline.hpp`  (+39/-0)
- `example/ck_tile/13_moe_sorting/README.md`  (+27/-0)
- `include/ck_tile/ops/fused_moe/pipeline/moe_sorting_problem.hpp`  (+23/-0)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.hpp`  (+20/-0)
- `example/ck_tile/13_moe_sorting/script/smoke_test.sh`  (+19/-0)
- `include/ck_tile/ops/fused_moe/pipeline/moe_sorting_policy.hpp`  (+15/-0)
- `include/ck_tile/ops/moe_sorting.hpp`  (+11/-0)
- `example/ck_tile/13_moe_sorting/CMakeLists.txt`  (+8/-0)
- `example/ck_tile/CMakeLists.txt`  (+1/-0)
- `include/ck_tile/host.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/13_moe_sorting/moe_sorting.cpp`**
```
auto create_args(int argc, char* argv[])
ck_tile::ArgParser arg_parser;
arg_parser.insert("v", "1", "weather do CPU validation or not")
.insert("pr_i", "int32", "index data type. (currently only int32 supported now)")
```

**`example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`**
```
constexpr ck_tile::index_t unroll_num = unroll_num_;                                    \
using ms_problem     = ck_tile::MoeSortingProblem<index_t, ms_weight_type, unroll_num>; \
using kernel         = ck_tile::MoeSortingKernel<ms_problem>;                           \
auto kargs           = kernel::MakeKargs(a);                                            \
```

**`example/ck_tile/13_moe_sorting/moe_sorting_api.hpp`**
```
struct moe_sorting_trait
std::string index_type;
std::string weight_type; // currently always float
struct moe_sorting_args : public ck_tile::MoeSortingHostArgs
```

**`include/ck_tile/host/reference/reference_moe_sorting.hpp`**
```
namespace ck_tile {
template <typename WeightType, typename IndexType = index_t>
CK_TILE_HOST void reference_moe_sorting(const HostTensor<IndexType>& topk_ids,
const HostTensor<WeightType>& weights,
```

**`include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`**
```
namespace ck_tile {
struct MoeSortingHostArgs
const void* p_topk_ids;
const void* p_weights;
```
