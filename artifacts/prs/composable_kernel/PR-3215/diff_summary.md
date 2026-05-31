# Diff summary

- **files changed:** 7
- **lines:** +61 / -14
- **kernel-ish files:** 6

## Files (by churn)

- `include/ck_tile/core/tensor/tile_elementwise.hpp`  (+30/-8)
- `include/ck_tile/core/numeric/bfloat16.hpp`  (+14/-1)
- `include/ck_tile/core/numeric/half.hpp`  (+6/-0)
- `test/ck_tile/grouped_gemm_multi_d/CMakeLists.txt`  (+4/-1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp`  (+2/-2)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+2/-2)
- `include/ck_tile/core/numeric/type_convert.hpp`  (+3/-0)

## Key added lines (kernel files)

**`include/ck_tile/core/numeric/bfloat16.hpp`**
```
using bf16x2_t = bfloat16_t __attribute__((ext_vector_type(2)));
using fp32x2_t = float __attribute__((ext_vector_type(2)));
template <bf16_rounding_mode rounding =
static_cast<bf16_rounding_mode>(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT)>
```

**`include/ck_tile/core/numeric/half.hpp`**
```
using fp32x2_t = float __attribute__((ext_vector_type(2)));
CK_TILE_HOST_DEVICE
constexpr fp16x2_t fp32x2_to_fp16x2(const fp32x2_t& x)
return fp16x2_t{float_to_fp16(x.x), float_to_fp16(x.y)};
```

**`include/ck_tile/core/numeric/type_convert.hpp`**
```
CK_TILE_TYPE_CONVERT(fp16x2_t, fp16x2, fp32x2_t, fp32x2)
CK_TILE_TYPE_CONVERT(bf16x2_t, bf16x2, fp32x2_t, fp32x2)
```

**`include/ck_tile/core/tensor/tile_elementwise.hpp`**
```
CK_TILE_DEVICE auto cast_tile_pkrtz_fp16_fp32(const InTensor& in_dstr_tensors)
template <typename OutDataType, typename InTensor>
CK_TILE_DEVICE auto cast_tile_pk_fp16bf16_fp32(const InTensor& in_dstr_tensors)
constexpr auto in_tile_dstr = InTensor::get_tile_distribution();
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp`**
```
return impl::cast_tile_pkrtz_fp16_fp32<PDataType>(
```
