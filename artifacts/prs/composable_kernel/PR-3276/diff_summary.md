# Diff summary

- **files changed:** 29
- **lines:** +1782 / -1134
- **kernel-ish files:** 26

## Files (by churn)

- `experimental/builder/include/ck_tile/builder/conv_factory.hpp`  (+0/-1050)
- `experimental/builder/include/ck_tile/builder/factory/conv_dispatcher.hpp`  (+196/-0)
- `experimental/builder/include/ck_tile/builder/factory/helpers/conv_tuning_params.hpp`  (+160/-0)
- `experimental/builder/include/ck_tile/builder/factory/helpers/conv_tensor_layout.hpp`  (+146/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_fwd_dl_factory.hpp`  (+138/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_fwd_v3_factory.hpp`  (+119/-0)
- `experimental/builder/test/unit_conv_tensor_layout.cpp`  (+119/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_fwd_large_tensor_factory.hpp`  (+117/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_fwd_xdl_factory.hpp`  (+114/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_fwd_wmma_factory.hpp`  (+113/-0)
- `experimental/builder/test/unit_conv_tuning_params.cpp`  (+90/-0)
- `experimental/builder/include/ck_tile/builder/factory/helpers/conv_tensor_type.hpp`  (+87/-0)
- `experimental/builder/test/unit_conv_tensor_type.cpp`  (+79/-0)
- `experimental/builder/include/ck_tile/builder/factory/helpers/conv_block_transfer.hpp`  (+73/-0)
- `experimental/builder/include/ck_tile/builder/conv_algorithm_concepts.hpp`  (+1/-38)

## Key added lines (kernel files)

**`experimental/builder/include/ck_tile/builder/conv_algorithm_concepts.hpp`**
```
{ t.m_xdl_per_wave_per_shuffle } -> std::convertible_to<size_t>;
```

**`experimental/builder/include/ck_tile/builder/conv_algorithm_limits.hpp`**
```
requires Value.scalar_per_vector > 0 && Value.m_xdl_per_wave_per_shuffle > 0 &&
Value.n_xdl_per_wave_per_shuffle > 0;
```

**`experimental/builder/include/ck_tile/builder/conv_builder.hpp`**
```
using Instance = decltype(factory::make_conv_instance<SIGNATURE, ALGORITHM, VERSION>());
```

**`experimental/builder/include/ck_tile/builder/factory/conv_dispatcher.hpp`**
```
namespace ck_tile::builder::factory {
template <typename T>
consteval bool IsXdlV3Algorithm()
return ConvAlgorithmDescriptor<T> && SpecifiesThreadBlock<T> && SpecifiesGridwiseXdlGemm<T> &&
```

**`experimental/builder/include/ck_tile/builder/factory/conv_fwd_dl_factory.hpp`**
```
namespace ck_tile::builder::factory {
template <ConvSignatureDescriptor auto SIGNATURE,
ConvAlgorithmDescriptor auto ALGORITHM,
StringLiteral VERSION>
```
