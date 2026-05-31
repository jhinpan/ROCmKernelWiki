# Diff summary

- **files changed:** 16
- **lines:** +2217 / -587
- **kernel-ish files:** 16

## Files (by churn)

- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_trload_default_policy.hpp`  (+1220/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp`  (+760/-0)
- `include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`  (+58/-501)
- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+62/-34)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp`  (+20/-22)
- `include/ck_tile/core/tensor/tensor_descriptor.hpp`  (+23/-5)
- `include/ck_tile/core/tensor/tensor_adaptor.hpp`  (+15/-7)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_selector.hpp`  (+14/-6)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp`  (+13/-3)
- `include/ck_tile/core/numeric/integral_constant.hpp`  (+11/-1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp`  (+4/-6)
- `include/ck_tile/host/device_prop.hpp`  (+6/-0)
- `example/ck_tile/01_fmha/fmha_bwd.hpp`  (+4/-1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`  (+3/-1)
- `include/ck_tile/ops/fmha.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
{F_trload},
{F_deterministic},
{F_trload}>;
const bool has_load_tr = ck_tile::is_load_tr_supported();
```

**`example/ck_tile/01_fmha/fmha_bwd.hpp`**
```
bool kIsDeterministic_,
bool kUseTrLoad_>
static constexpr bool kUseTrLoad       = kUseTrLoad_;
```

**`include/ck_tile/core/numeric/integral_constant.hpp`**
```
template <typename T>
struct is_constant : std::false_type
template <auto v>
struct is_constant<constant<v>> : std::true_type
```

**`include/ck_tile/core/tensor/tensor_adaptor.hpp`**
```
template <index_t Internal = 0>
static_for<0,
Internal ? std::min(Internal, get_num_of_transform()) : get_num_of_transform(),
1>{}([&](auto itran) {
```

**`include/ck_tile/core/tensor/tensor_descriptor.hpp`**
```
template <index_t Internal = 0>
return Base::template get_top_dimension_safe_vector_length_strides<Internal>(
constexpr index_t first_dim_length = []() {
if constexpr(is_constant_v<remove_cvref_t<decltype(element_space_size)>>)
```
