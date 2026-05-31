# Diff summary

- **files changed:** 20
- **lines:** +1924 / -1410
- **kernel-ish files:** 19

## Files (by churn)

- `include/ck_tile/ops/flatmm/block/uk/flatmm_uk_gfx9_32x512x128_1x1x1_16x16x16.inc`  (+728/-471)
- `include/ck_tile/ops/flatmm/block/uk/flatmm_sn_uk_gfx9_32x128x512_1x4x1_16x16x16_itl.inc`  (+523/-567)
- `include/ck_tile/ops/flatmm/block/flatmm_32x512x128_1x4x1_16x16x32.hpp`  (+282/-234)
- `include/ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_flatmm_uk.hpp`  (+98/-36)
- `example/ck_tile/15_fused_moe/main.cpp`  (+55/-52)
- `include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`  (+75/-0)
- `example/ck_tile/15_fused_moe/instances/fused_moegemm_api.cpp`  (+56/-4)
- `include/ck_tile/host/reference/reference_fused_moe.hpp`  (+25/-16)
- `example/ck_tile/15_fused_moe/instances/fused_moegemm_api_internal.hpp`  (+25/-15)
- `example/ck_tile/15_fused_moe/instances/fused_moegemm_fp16_m32.cpp`  (+13/-1)
- `example/ck_tile/15_fused_moe/instances/fused_moegemm_bf16_m32.cpp`  (+12/-1)
- `include/ck_tile/ops/fused_moe/kernel/fused_moegemm_kernel.hpp`  (+7/-4)
- `include/ck_tile/ops/flatmm/block/uk/flatmm_sn_uk_gfx9_32x128x512_1x4x1_16x16x16.inc`  (+9/-0)
- `example/ck_tile/15_fused_moe/instances/fused_moegemm_api_traits.hpp`  (+4/-2)
- `example/ck_tile/15_fused_moe/fused_moe.hpp`  (+3/-2)

## Key added lines (kernel files)

**`example/ck_tile/15_fused_moe/fused_moe.hpp`**
```
ck_tile::index_t intermediate_size; // n / TP, for Gate. and Up, Down is also this value
int activation;  // 0:gelu, 1:silu
int gate_only;   // 0:g1u0, 1:g1u1
```

**`example/ck_tile/15_fused_moe/fused_moegemm.hpp`**
```
int activation;  // 0:gelu, 1:silu
int gate_only;   // 0:g1u0, 1:g1u1
```

**`example/ck_tile/15_fused_moe/instances/fused_moe_api.cpp`**
```
t.activation,
```

**`example/ck_tile/15_fused_moe/instances/fused_moegemm_api.cpp`**
```
t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.block_m == 32 && t.gate_only == 1 && t.activation
constexpr ck_tile::index_t act_ = 0;
constexpr ck_tile::index_t go_  = 1;
using t_ = fmoe_<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float, float, float, float, S<32, 512, 128, 128>, S<
```

**`example/ck_tile/15_fused_moe/instances/fused_moegemm_api_internal.hpp`**
```
constexpr auto get_activation_ = []() {
if constexpr(Ts_::Activation == 0)
return ck_tile::element_wise::FastGeluAsm{};
return ck_tile::element_wise::Silu{};
```
