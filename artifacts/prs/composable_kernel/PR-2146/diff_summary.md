# Diff summary

- **files changed:** 15
- **lines:** +913 / -151
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+232/-20)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_util.hpp`  (+193/-16)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_tileloop.cpp`  (+174/-0)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+72/-70)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+77/-21)
- `include/ck_tile/host/stream_utils.hpp`  (+45/-0)
- `test/ck_tile/grouped_gemm/test_grouped_gemm.cpp`  (+19/-11)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_ut_cases.inc`  (+28/-2)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp`  (+23/-1)
- `include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`  (+17/-1)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+14/-3)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+7/-5)
- `include/ck_tile/core/utility/type_traits.hpp`  (+11/-0)
- `example/ck_tile/17_grouped_gemm/CMakeLists.txt`  (+1/-1)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`  (+0/-0)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
void* kargs_ptr)
const auto Run = [&](const auto has_hot_loop_,
const auto tail_number_,
const auto memory_operation_) {
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`**
```
.insert("group_count", "8", "group count.")
.insert("kbatch", "1", "kbatch for SplitK");
inline std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
return gemm_descs.size() * sizeof(ck_tile::GemmTransKernelArg);
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm_tileloop.cpp`**
```
template <typename ALayout, typename BLayout, typename CLayout>
float grouped_gemm_tileloop(const ck_tile::stream_config& s,
const ck_tile::index_t num_groups,
void* kargs_ptr,
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`**
```
template <typename ALayout, typename BLayout, typename CLayout, bool Persistent>
float ave_time = 0;
if constexpr(!Persistent)
ave_time = grouped_gemm<ALayout, BLayout, CLayout>(
```

**`include/ck_tile/core/utility/type_traits.hpp`**
```
template <typename Test, template <typename...> class RefTemplate>
struct is_specialization_of : std::false_type
template <template <typename...> class RefTemplate, typename... Args>
struct is_specialization_of<RefTemplate<Args...>, RefTemplate> : std::true_type
```
