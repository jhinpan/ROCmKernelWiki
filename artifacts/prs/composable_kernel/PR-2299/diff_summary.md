# Diff summary

- **files changed:** 10
- **lines:** +233 / -18
- **kernel-ish files:** 9

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+105/-0)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+31/-6)
- `include/ck_tile/core/utility/type_traits.hpp`  (+30/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+16/-6)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+13/-3)
- `test/ck_tile/gemm/test_gemm_pipeline_persistent.cpp`  (+16/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_kernel_types.hpp`  (+9/-0)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+4/-2)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+4/-1)
- `test/ck_tile/gemm/CMakeLists.txt`  (+5/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
typename CLayout,
bool Persistent>
if constexpr(Persistent)
std::cout << "WARNING: Ignoring persistent kernel option for basic gemm." << std::endl;
```

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
.insert("init", "0", "0:random, 1:linear, 2:constant(1)")
.insert("persistent", "0", "0:non-persistent, 1:persistent");
typename CLayout,
bool Persistent = false>
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
int n_repeat,
bool persistent)
float ave_time;
if(persistent)
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
typename CLayout,
bool Persistent>
GemmConfig::UseStructuredSparsity,
Persistent>;
```

**`include/ck_tile/core/utility/type_traits.hpp`**
```
namespace detail {
template <bool IsWithinBounds, std::size_t Idx, typename Tuple, typename DefaultType>
struct tuple_element_or_default_dispatch
using type = DefaultType;
```
