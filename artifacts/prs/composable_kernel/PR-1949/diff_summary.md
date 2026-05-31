# Diff summary

- **files changed:** 15 (diff was byte-capped; summary is partial)
- **lines:** +5615 / -44
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_preshuffle.hpp`  (+1681/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_dequant_v3.hpp`  (+930/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_dequant_v1.hpp`  (+547/-0)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_pk_i4.cpp`  (+525/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_b_preshuffle.hpp`  (+517/-0)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_pk_i4.cpp`  (+488/-0)
- `example/01_gemm/gemm_xdl_fp8_pk_i4_bpreshuffle_v3.cpp`  (+350/-0)
- `example/01_gemm/gemm_xdl_fp8_pk_i4_v3.cpp`  (+329/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_selector.hpp`  (+100/-41)
- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+67/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_v2.hpp`  (+35/-1)
- `example/01_gemm/common.hpp`  (+23/-0)
- `example/65_gemm_multiply_multiply/CMakeLists.txt`  (+11/-1)
- `example/01_gemm/CMakeLists.txt`  (+10/-0)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8.cpp`  (+2/-1)

## Key added lines (kernel files)

**`example/01_gemm/common.hpp`**
```
float i4_to_f32_gfx9(uint8_t i4)
static std::unordered_map<uint8_t, float> u = {{0b1000, -0.5000f},
{0b1001, -0.4375f},
{0b1010, -0.3750f},
```

**`example/01_gemm/gemm_xdl_fp8_pk_i4_bpreshuffle_v3.cpp`**
```
using F8  = ck::f8_t;
using I4  = ck::pk_i4_t;
using F16 = ck::half_t;
using F32 = float;
```

**`example/01_gemm/gemm_xdl_fp8_pk_i4_v3.cpp`**
```
using F8  = ck::f8_t;
using I4  = ck::pk_i4_t;
using F16 = ck::half_t;
using F32 = float;
```

**`example/65_gemm_multiply_multiply/moe_gemm1_xdl_pk_i4.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using I4  = ck::pk_i4_t;
using F16 = ck::half_t;
```

**`example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8.cpp`**
```
if(tile_off < token_per_tile && tokenid < tokens * topk)
```
