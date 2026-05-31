# Diff summary

- **files changed:** 7
- **lines:** +806 / -90
- **kernel-ish files:** 6

## Files (by churn)

- `include/ck_tile/core/numeric/pk_fp4.hpp`  (+324/-0)
- `include/ck_tile/core/numeric/mxfp_convert.hpp`  (+213/-0)
- `include/ck_tile/core/numeric/numeric.hpp`  (+88/-90)
- `test/ck_tile/data_type/test_pk_fp4.cpp`  (+162/-0)
- `include/ck_tile/core/numeric/type_convert.hpp`  (+16/-0)
- `include/ck_tile/core.hpp`  (+2/-0)
- `test/ck_tile/data_type/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`include/ck_tile/core/numeric/mxfp_convert.hpp`**
```
namespace ck_tile {
template <typename T>
struct numeric_utils : numeric_traits<T>
using traits   = numeric_traits<T>;
```

**`include/ck_tile/core/numeric/numeric.hpp`**
```
attr_ bool operator==(const type_& x, const type_& y)                                  \
{                                                                                      \
return std::abs(static_cast<float>(x) - static_cast<float>(y)) <                   \
static_cast<float>(numeric<type_>::epsilon());                              \
```

**`include/ck_tile/core/numeric/pk_fp4.hpp`**
```
namespace ck_tile {
using fp32_t   = float;
using fp32x2_t = float __attribute__((ext_vector_type(2)));
using fp16x2_t = _Float16 __attribute__((ext_vector_type(2)));
```

**`include/ck_tile/core/numeric/type_convert.hpp`**
```
} // namespace ck_tile
namespace ck_tile {
CK_TILE_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp32x2_t, fp32x2)
CK_TILE_TYPE_CONVERT(fp32x2_t, fp32x2, pk_fp4_t, pk_fp4)
```

**`test/ck_tile/data_type/test_pk_fp4.cpp`**
```
using ck_tile::bf16_t;
using ck_tile::bf16x2_t;
using ck_tile::fp16_t;
using ck_tile::fp16x2_t;
```
