# Diff summary

- **files changed:** 8
- **lines:** +419 / -86
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck_tile/core/numeric/pk_fp4.hpp`  (+99/-64)
- `test/ck_tile/data_type/test_mx_scale.cpp`  (+162/-0)
- `include/ck_tile/core/numeric/e8m0.hpp`  (+102/-0)
- `include/ck_tile/core/numeric/type_convert.hpp`  (+31/-10)
- `include/ck_tile/core/numeric/mxfp_convert.hpp`  (+16/-11)
- `include/ck_tile/host/host_tensor.hpp`  (+7/-1)
- `include/ck_tile/core.hpp`  (+1/-0)
- `test/ck_tile/data_type/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`include/ck_tile/core/numeric/e8m0.hpp`**
```
namespace ck_tile {
struct e8m0_bexp_t
using raw_type = uint8_t;
using type     = raw_type;
```

**`include/ck_tile/core/numeric/mxfp_convert.hpp`**
```
using raw_type = typename traits::bitwise_type;
static constexpr raw_type get_exponent(raw_type x)
static constexpr raw_type get_exponent(const T& x)
return get_exponent(bit_cast<raw_type>(x));
```

**`include/ck_tile/core/numeric/pk_fp4.hpp`**
```
CK_TILE_HOST_DEVICE constexpr uint8_t float_to_e2m1(float x, float scale = 1.f);
CK_TILE_HOST_DEVICE explicit constexpr pk_float4_e2m1_t(float init, float scale = 1.f)
: data{float_to_e2m1(init, scale)}
CK_TILE_HOST_DEVICE constexpr float to_float(float scale = 1.f) const;
```

**`include/ck_tile/core/numeric/type_convert.hpp`**
```
template <typename Y, typename X>
CK_TILE_HOST_DEVICE constexpr Y scaled_type_convert(X x, float scale);
template <>                                                                           \
CK_TILE_HOST_DEVICE constexpr dtype_ scaled_type_convert<dtype_, stype_>(stype_ x,    \
```

**`include/ck_tile/host/host_tensor.hpp`**
```
void SetZero()
if constexpr(std::is_same_v<T, e8m0_t>)
std::fill(mData.begin(), mData.end(), e8m0_t{1.f});
std::fill(mData.begin(), mData.end(), 0);
```
