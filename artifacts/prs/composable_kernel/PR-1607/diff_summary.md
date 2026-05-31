# Diff summary

- **files changed:** 14
- **lines:** +1318 / -0
- **kernel-ish files:** 10

## Files (by churn)

- `example/ck_tile/06_permute/alternative_impl/matrix_core_swizzle_kernel.hpp`  (+413/-0)
- `example/ck_tile/06_permute/permute.cpp`  (+411/-0)
- `include/ck_tile/ops/permute/kernel/generic_permute_kernel.hpp`  (+169/-0)
- `example/ck_tile/06_permute/alternative_impl/matrix_core_swizzle.cpp`  (+98/-0)
- `include/ck_tile/host/reference/reference_permute.hpp`  (+57/-0)
- `example/ck_tile/06_permute/README.md`  (+46/-0)
- `example/ck_tile/06_permute/script/smoke_test.sh`  (+34/-0)
- `include/ck_tile/ops/permute/pipeline/generic_petmute_problem.hpp`  (+28/-0)
- `example/ck_tile/06_permute/alternative_impl/matrix_core_swizzle.hpp`  (+20/-0)
- `example/ck_tile/06_permute/permute.hpp`  (+19/-0)
- `example/ck_tile/06_permute/CMakeLists.txt`  (+13/-0)
- `include/ck_tile/ops/permute.hpp`  (+8/-0)
- `example/ck_tile/CMakeLists.txt`  (+1/-0)
- `include/ck_tile/host.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/06_permute/alternative_impl/matrix_core_swizzle.cpp`**
```
float matrix_core_swizzle(matrix_core_swizzle_traits t,
matrix_core_swizzle_args a,
const ck_tile::stream_config& s)
if(t.data_type.compare("fp16") == 0)
```

**`example/ck_tile/06_permute/alternative_impl/matrix_core_swizzle.hpp`**
```
struct matrix_core_swizzle_traits
std::string data_type; // fp16 only
std::string inst;      // 32x32x8, 16x16x16
std::string permute;   //
```

**`example/ck_tile/06_permute/alternative_impl/matrix_core_swizzle_kernel.hpp`**
```
enum class matrix_core_inst_enum
MFMA_32x32x8_F16  = 0,
MFMA_16x16x16_F16 = 1,
namespace detail {
```

**`example/ck_tile/06_permute/permute.cpp`**
```
namespace detail {
template <int bytes>
struct to_integer_type;
template <>
```

**`example/ck_tile/06_permute/permute.hpp`**
```
struct permute_traits
std::string data_type;
using permute_args = ck_tile::GenericPermuteHostArgs;
float permute(permute_traits, permute_args, const ck_tile::stream_config&);
```
