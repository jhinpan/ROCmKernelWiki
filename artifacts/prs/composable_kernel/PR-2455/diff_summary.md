# Diff summary

- **files changed:** 10 (diff was byte-capped; summary is partial)
- **lines:** +3019 / -1894
- **kernel-ish files:** 9

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_gufusion_v3.hpp`  (+747/-567)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_gufusion_v1.hpp`  (+0/-919)
- `example/67_gemm_microscaling/moe_gemm2_xdl_mx_fp4_bpreshuffle.cpp`  (+584/-0)
- `example/67_gemm_microscaling/moe_gemm1_xdl_mx_fp4_bpreshuffle.cpp`  (+574/-0)
- `example/67_gemm_microscaling/moe_gemm1_xdl_mx_fp4.cpp`  (+548/-0)
- `example/67_gemm_microscaling/moe_gemm2_xdl_mx_fp4.cpp`  (+542/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_v1.hpp`  (+0/-358)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_selector.hpp`  (+2/-47)
- `example/67_gemm_microscaling/CMakeLists.txt`  (+21/-2)
- `example/67_gemm_microscaling/moe_gemm2_xdl_mx_fp4_bns.cpp`  (+1/-1)

## Key added lines (kernel files)

**`example/67_gemm_microscaling/moe_gemm1_xdl_mx_fp4.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F4              = ck::f4x2_pk_t;
using F16             = ck::half_t;
```

**`example/67_gemm_microscaling/moe_gemm1_xdl_mx_fp4_bpreshuffle.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F4              = ck::f4x2_pk_t;
using F16             = ck::half_t;
```

**`example/67_gemm_microscaling/moe_gemm2_xdl_mx_fp4.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F4              = ck::f4x2_pk_t;
using F16             = ck::half_t;
```

**`example/67_gemm_microscaling/moe_gemm2_xdl_mx_fp4_bns.cpp`**
```
2,    4,   S<1, 4, 1, 64>, S<2, 1, 1, 1>,
```

**`example/67_gemm_microscaling/moe_gemm2_xdl_mx_fp4_bpreshuffle.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F4              = ck::f4x2_pk_t;
using F16             = ck::half_t;
```
