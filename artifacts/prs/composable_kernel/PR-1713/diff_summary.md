# Diff summary

- **files changed:** 16
- **lines:** +1186 / -0
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+310/-0)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_util.hpp`  (+282/-0)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+191/-0)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+151/-0)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+53/-0)
- `example/ck_tile/17_grouped_gemm/utils.hpp`  (+38/-0)
- `include/ck_tile/core/utility/amd_address_space.hpp`  (+37/-0)
- `include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`  (+36/-0)
- `test/ck_tile/grouped_gemm/test_grouped_gemm.cpp`  (+29/-0)
- `example/ck_tile/17_grouped_gemm/README.md`  (+25/-0)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_ut_cases.inc`  (+25/-0)
- `test/ck_tile/grouped_gemm/CMakeLists.txt`  (+4/-0)
- `example/ck_tile/17_grouped_gemm/CMakeLists.txt`  (+2/-0)
- `example/ck_tile/CMakeLists.txt`  (+1/-0)
- `include/ck_tile/ops/gemm.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
namespace {
struct GroupedGemmKernelParam
static const bool kPadM        = false;
static const bool kPadN        = false;
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`**
```
template <typename DataType>
struct GemmBasicTypeConfig;
template <>
struct GemmBasicTypeConfig<ck_tile::half_t>
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`**
```
template <typename ALayout, typename BLayout, typename CLayout>
float invoke_gemm(int n_warmup,
int n_repeat,
int group_count,
```

**`example/ck_tile/17_grouped_gemm/utils.hpp`**
```
template <typename TLayout>
constexpr auto
f_host_tensor_descriptor(std::size_t row, std::size_t col, std::size_t stride, TLayout layout)
using namespace ck_tile::literals;
```

**`include/ck_tile/core/utility/amd_address_space.hpp`**
```
namespace ck_tile {
template <typename T>
__device__ T* cast_pointer_to_generic_address_space(T CK_CONSTANT_ADDRESS_SPACE* p)
return (T*)p; // NOLINT(old-style-cast)
```
