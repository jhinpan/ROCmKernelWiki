# Diff summary

- **files changed:** 11
- **lines:** +426 / -899
- **kernel-ish files:** 9

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v3.hpp`  (+290/-380)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply_wp.hpp`  (+0/-389)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v2.hpp`  (+41/-27)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v1.hpp`  (+32/-21)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_wp/CMakeLists.txt`  (+0/-48)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply_wp/f8_f8_bf16/device_gemm_multiply_multiply_wp_xdl_f8_f8_bf16_mk_mfma_mn.hpp`  (+22/-22)
- `include/ck/utility/blkgemmpipe_scheduler.hpp`  (+20/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle.hpp`  (+9/-7)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_bpreshuffle.cpp`  (+4/-5)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp`  (+5/-0)
- `CHANGELOG.md`  (+3/-0)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_bpreshuffle.cpp`**
```
16,   16,
2,    1,   S<1, 32, 1, 8>, S<8, 8, 1>,
ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, FP8>;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v1.hpp`**
```
using Base::KGroup;
constexpr index_t K2 = KPack / KGroup;
constexpr index_t K0 = KRepeat * KGroup;
static_for<0, KGroup, 1>{}([&](auto kg0) {
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v2.hpp`**
```
using Base::KGroup;
constexpr index_t K2 = KPack / KGroup;
constexpr index_t K0 = KRepeat * KGroup;
static_for<0, KGroup, 1>{}([&](auto kg0) {
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v3.hpp`**
```
template <typename T>
constexpr auto compute_stage_loads(T total_loads, T stages)
return std::make_pair((total_loads + stages - 1) / stages, // ceil
total_loads / stages                 // floor
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp`**
```
static constexpr index_t KGroup =
((MPerXDL == 16 && MPerXDL == 16 && xdlops_gemm.KPerXdlops == 128) ||
(MPerXDL == 32 && MPerXDL == 32 && xdlops_gemm.KPerXdlops == 64))
```
