# Diff summary

- **files changed:** 33
- **lines:** +161 / -152
- **kernel-ish files:** 32

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_batch_prefill_kernel.hpp`  (+31/-30)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_pagedkv_kernel.hpp`  (+26/-24)
- `Jenkinsfile`  (+11/-11)
- `include/ck_tile/core/algorithm/coordinate_transform.hpp`  (+9/-8)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_kernel.hpp`  (+9/-8)
- `include/ck_tile/core/numeric/math.hpp`  (+1/-13)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+8/-6)
- `include/ck_tile/core/tensor/tensor_descriptor.hpp`  (+8/-4)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp`  (+5/-5)
- `include/ck_tile/core/container/sequence.hpp`  (+5/-4)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+5/-4)
- `include/ck_tile/core/algorithm/space_filling_curve.hpp`  (+4/-4)
- `include/ck_tile/core/utility/functional_with_tuple.hpp`  (+4/-4)
- `include/ck_tile/core/utility/unary_element_function.hpp`  (+5/-2)
- `include/ck_tile/core/tensor/tile_distribution.hpp`  (+3/-3)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/fmha_fwd_runner.hpp`**
```
return ck_tile::make_composes(ck_tile::saturates<ck_tile::fp8_t>{},
ck_tile::scales{scale_o_host});
```

**`include/ck_tile/core/algorithm/coordinate_transform.hpp`**
```
decltype(make_tuple(container_reduce(LowLengths{}, multiplies<>{}, number<1>{})));
up_lengths_{make_tuple(container_reduce(low_lengths, multiplies<>{}, I1))}
decltype(container_reverse_exclusive_scan(LowLengths{}, multiplies<>{}, number<1>{}));
decltype(make_tuple(container_reduce(LowLengths{}, multiplies<>{}, number<1>{})));
```

**`include/ck_tile/core/algorithm/space_filling_curve.hpp`**
```
reduce_on_sequence(TensorLengths{}, multiplies<>{}, number<1>{});
reduce_on_sequence(ScalarsPerAccess{}, multiplies<>{}, number<1>{});
return reduce_on_sequence(TensorLengths{}, multiplies<>{}, number<1>{}) / ScalarPerVector;
container_reverse_exclusive_scan(ordered_access_lengths, multiplies<>{}, number<1>{});
```

**`include/ck_tile/core/container/sequence.hpp`**
```
static_assert(
container_reduce(pick_sequence_elements_by_mask(Seq{}, Mask{}), multiplies<>{}, 1) %
SliceSize ==
"slice size can't evenly divide input sizes");
```

**`include/ck_tile/core/numeric/math.hpp`**
```
scales(Scale) -> scales<Scale>;
```
