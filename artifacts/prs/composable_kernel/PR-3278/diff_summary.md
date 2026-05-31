# Diff summary

- **files changed:** 30
- **lines:** +2483 / -87
- **kernel-ish files:** 27

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3_b_preshuffle.hpp`  (+609/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`  (+543/-7)
- `example/65_gemm_multiply_multiply/run_gemm_multiply_multiply_wp_example.inc`  (+246/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_thread_tiles_preshuffle.hpp`  (+138/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+72/-48)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_wp/device_gemm_multiply_multiply_wp_wmma_f8_f8_bf16_mk_wmma_mn.hpp`  (+105/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_wp/device_gemm_multiply_multiply_wp_wmma_f8_f8_f16_mk_wmma_mn.hpp`  (+105/-0)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_wmma_fp8_bpreshuffle.cpp`  (+94/-0)
- `example/65_gemm_multiply_multiply/common.hpp`  (+82/-0)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_wmma_fp16_bpreshuffle.cpp`  (+82/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+38/-19)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply_wp.hpp`  (+55/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_wp/device_gemm_multiply_multiply_wp_wmma_f8_f8_bf16_mk_wmma_mn_default_instance_p1.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_wp/device_gemm_multiply_multiply_wp_wmma_f8_f8_bf16_mk_wmma_mn_default_instance_p2.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_wp/device_gemm_multiply_multiply_wp_wmma_f8_f8_bf16_mk_wmma_mn_default_instance_p3.cpp`  (+33/-0)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/common.hpp`**
```
struct MultiplyMultiply
template <typename E, typename C, typename D0, typename D1>
__host__ __device__ constexpr void
operator()(E& e, const C& c, const D0& d0, const D1& d1) const;
```

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_wmma_fp16_bpreshuffle.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
```

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_wmma_fp8_bpreshuffle.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F8   = ck::f8_t;
using F16  = ck::half_t;
```

**`example/65_gemm_multiply_multiply/run_gemm_multiply_multiply_wp_example.inc`**
```
int run_gemm_example(int argc, char* argv[])
bool do_verification = true;
int init_method      = 1;
bool time_kernel     = false;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp`**
```
bool TransposeC = false,
bool BSkipLDS   = false>
TransposeC,
BSkipLDS>{};
```
