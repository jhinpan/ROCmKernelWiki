# Diff summary

- **files changed:** 67 (diff was byte-capped; summary is partial)
- **lines:** +2770 / -763
- **kernel-ish files:** 65

## Files (by churn)

- `experimental/builder/test/impl/conv_algorithm_types.hpp`  (+264/-70)
- `experimental/builder/include/ck_tile/builder/conv_algorithm_concepts.hpp`  (+175/-71)
- `experimental/builder/test/utils/ckb_conv_test_configs.hpp`  (+162/-64)
- `experimental/builder/test/unit_conv_tensor_layout.cpp`  (+71/-71)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_dl_factory.hpp`  (+131/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_dispatcher.hpp`  (+68/-61)
- `experimental/builder/include/ck_tile/builder/factory/conv_algorithms.hpp`  (+128/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_two_stage_wmma_v3_factory.hpp`  (+111/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_two_stage_xdl_factory.hpp`  (+111/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_multi_d_wmma_v3_factory.hpp`  (+110/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_wmma_factory.hpp`  (+109/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_wmma_v3_factory.hpp`  (+109/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_xdl_v3_factory.hpp`  (+108/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_multi_d_xdl_factory.hpp`  (+103/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_xdl_factory.hpp`  (+103/-0)

## Key added lines (kernel files)

**`experimental/builder/include/ck_tile/builder/conv_algorithm_concepts.hpp`**
```
template <typename T>
concept SizeType = std::unsigned_integral<std::remove_cvref_t<T>>;
{ t.block_size } -> SizeType;
{ t.tile_size.m } -> SizeType;
```

**`experimental/builder/include/ck_tile/builder/conv_algorithm_limits.hpp`**
```
concept AccessOrderLimits3D = requires {
(Value[2] >= 0 && Value[2] < 3) && (Value.Size() == 3));
template <auto Value>
concept AccessOrderLimits4D = requires {
```

**`experimental/builder/include/ck_tile/builder/conv_signature_concepts.hpp`**
```
template <auto Sig>
concept Is3D = requires {
requires Sig.spatial_dim == 3;
requires ConvInputLayout3D<Sig.input.config.layout>;
```

**`experimental/builder/include/ck_tile/builder/factory/conv_algorithms.hpp`**
```
namespace ck_tile::builder::factory {
template <typename T, size_t ThreadClusterRank = 3>
concept TileTransferParameters =
SpecifiesBlockTransfer<T, ThreadClusterRank> && SpecifiesLdsTransfer<T> &&
```

**`experimental/builder/include/ck_tile/builder/factory/conv_bwd_weight_dl_factory.hpp`**
```
namespace ck_tile::builder::factory {
template <ConvSignatureDescriptor auto SIGNATURE,
ConvAlgorithmDescriptor auto ALGORITHM,
StringLiteral VERSION>
```
