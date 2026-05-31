# Diff summary

- **files changed:** 55
- **lines:** +1431 / -92
- **kernel-ish files:** 54

## Files (by churn)

- `experimental/builder/include/ck_tile/builder/factory/helpers/ck_tile/conv_tile_tensor_layout.hpp`  (+200/-0)
- `experimental/builder/include/ck_tile/builder/factory/helpers/ck_tile/conv_tile_tuning_params.hpp`  (+158/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_tile_factory.hpp`  (+131/-0)
- `experimental/builder/test/impl/conv_algorithm_types.hpp`  (+118/-0)
- `experimental/builder/include/ck_tile/builder/factory/helpers/ck_tile/conv_tile_kernel_directions.hpp`  (+88/-0)
- `experimental/builder/include/ck_tile/builder/factory/helpers/ck_tile/conv_tile_tensor_type.hpp`  (+87/-0)
- `experimental/builder/include/ck_tile/builder/conv_algorithm_concepts.hpp`  (+84/-1)
- `experimental/builder/test/utils/ckb_conv_tile_test_configs.hpp`  (+85/-0)
- `experimental/builder/include/ck_tile/builder/factory/helpers/ck_tile/conv_tile_elementwise_op.hpp`  (+62/-0)
- `experimental/builder/test/conv/ck_tile/test_ckb_conv_bwd_data_2d_fp16_v3.cpp`  (+52/-0)
- `experimental/builder/test/conv/ck_tile/test_ckb_conv_bwd_weight_2d_fp16_v3.cpp`  (+52/-0)
- `experimental/builder/test/conv/ck_tile/test_ckb_conv_fwd_2d_fp16_v3.cpp`  (+52/-0)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+21/-16)
- `include/ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp`  (+37/-0)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+19/-17)

## Key added lines (kernel files)

**`experimental/builder/include/ck_tile/builder/conv_algorithm_concepts.hpp`**
```
template <typename T>
concept TileThreadBlockDescriptor = requires(T t) {
{ t.tile_size.m } -> std::convertible_to<size_t>;
{ t.tile_size.n } -> std::convertible_to<size_t>;
```

**`experimental/builder/include/ck_tile/builder/conv_algorithm_limits.hpp`**
```
template <auto Value>
concept TileInputOutputVectorTransferLimits =
requires { requires Value.a > 0 && Value.b > 0 && Value.c > 0; };
```

**`experimental/builder/include/ck_tile/builder/factory/conv_dispatcher.hpp`**
```
template <typename T>
consteval bool IsTileAlgorithm()
return ConvAlgorithmDescriptor<T> && SpecifiesTileThreadBlock<T> && SpecifiesTileTransfer<T> &&
SpecifiesTileConvSpecialization<T> && SpecifiesTileBlockGemm<T> &&
```

**`experimental/builder/include/ck_tile/builder/factory/conv_tile_factory.hpp`**
```
namespace ck_tile::builder::factory {
template <ConvSignatureDescriptor auto SIGNATURE,
ConvAlgorithmDescriptor auto ALGORITHM,
StringLiteral VERSION>
```

**`experimental/builder/include/ck_tile/builder/factory/helpers/ck_tile/conv_tile_block_transfer.hpp`**
```
namespace ck_tile::builder::factory::internal {
struct TileScalarPerVector
size_t a = 0;
size_t b = 0;
```
