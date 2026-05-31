# Diff summary

- **files changed:** 27
- **lines:** +939 / -262
- **kernel-ish files:** 26

## Files (by churn)

- `experimental/builder/include/ck_tile/builder/testing/conv/bwd_weight_ck.hpp`  (+276/-0)
- `experimental/builder/include/ck_tile/builder/testing/conv/reference.hpp`  (+137/-0)
- `experimental/builder/test/conv/ck_tile/test_ckb_conv_bwd_weight_2d_fp16_v3.cpp`  (+70/-24)
- `experimental/builder/include/ck_tile/builder/testing/conv/ck_tile.hpp`  (+61/-31)
- `experimental/builder/include/ck_tile/builder/testing/conv_fwd_reference.hpp`  (+0/-88)
- `experimental/builder/include/ck_tile/builder/testing/conv/bwd_weight.hpp`  (+71/-0)
- `experimental/builder/include/ck_tile/builder/testing/conv/fwd.hpp`  (+69/-0)
- `experimental/builder/include/ck_tile/builder/testing/conv/args.hpp`  (+7/-57)
- `experimental/builder/include/ck_tile/builder/testing/testing.hpp`  (+59/-3)
- `experimental/builder/test/conv/ck/test_ckb_conv_bwd_weight_xdl_cshuffle_v3.cpp`  (+55/-4)
- `experimental/builder/include/ck_tile/builder/testing/conv/fwd_ck.hpp`  (+29/-29)
- `experimental/builder/test/testing_utils.hpp`  (+32/-0)
- `experimental/builder/test/testing_utils.cpp`  (+18/-0)
- `experimental/builder/test/test_testing_utils.cpp`  (+17/-0)
- `experimental/builder/test/conv/ck/test_ckb_conv_fwd_2d_fp16.cpp`  (+8/-5)

## Key added lines (kernel files)

**`experimental/builder/include/ck_tile/builder/testing/conv/args.hpp`**
```
requires ValidConvSignature<SIGNATURE>
```

**`experimental/builder/include/ck_tile/builder/testing/conv/bwd_weight.hpp`**
```
namespace ck_tile::builder::test {
template <auto SIGNATURE>
requires ValidConvSignature<SIGNATURE> && ConvDirectionIsBackwardWeight<SIGNATURE>
struct Inputs<SIGNATURE>
```

**`experimental/builder/include/ck_tile/builder/testing/conv/bwd_weight_ck.hpp`**
```
namespace ck_tile::builder::test {
namespace detail {
template <typename Conv,
auto SIGNATURE,
```

**`experimental/builder/include/ck_tile/builder/testing/conv/ck_tile.hpp`**
```
requires ValidConvSignature<SIGNATURE>;
template <auto SIGNATURE, typename InDataType, typename WeiDataType, typename OutDataType>
[[nodiscard]] RunResult run(CkTileConvInstance<SIGNATURE> auto& conv,
const Args<SIGNATURE>& args,
```

**`experimental/builder/include/ck_tile/builder/testing/conv/fwd.hpp`**
```
namespace ck_tile::builder::test {
template <auto SIGNATURE>
requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
struct Inputs<SIGNATURE>
```
