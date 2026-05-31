# Diff summary

- **files changed:** 24
- **lines:** +613 / -331
- **kernel-ish files:** 24

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`  (+160/-140)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`  (+144/-100)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_wmma_cshuffle_v3.hpp`  (+112/-39)
- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_thread_tiles.hpp`  (+90/-8)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp`  (+28/-19)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+21/-4)
- `include/ck/tensor_operation/gpu/warp/wmma_gemm.hpp`  (+14/-0)
- `example/01_gemm/gemm_wmma_fp8_v3.cpp`  (+5/-5)
- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_wave_tiles.hpp`  (+4/-2)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_f16_f16_f16/device_gemm_wmma_universal_f16_f16_f16_km_kn_mn.hpp`  (+3/-1)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_f16_f16_f16/device_gemm_wmma_universal_f16_f16_f16_km_nk_mn.hpp`  (+3/-1)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_f16_f16_f16/device_gemm_wmma_universal_f16_f16_f16_mk_kn_mn.hpp`  (+3/-1)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_f16_f16_f16/device_gemm_wmma_universal_f16_f16_f16_mk_nk_mn.hpp`  (+3/-1)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp`  (+3/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_f16_f8_f16/device_gemm_wmma_universal_f16_f8_f16_km_kn_mn.hpp`  (+2/-1)

## Key added lines (kernel files)

**`example/01_gemm/gemm_wmma_fp8_v3.cpp`**
```
using ALayout = Col;
16, 16, // AK1, BK1
S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>,
1, 4, 16, 0,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp`**
```
index_t KInner,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp`**
```
index_t KInner,
static constexpr auto I6 = Number<6>{};
static constexpr auto wmma_gemm = WmmaGemm<ComputeTypeA,
ComputeTypeB,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`**
```
index_t KInner,
index_t KInner,
constexpr index_t KPerWaveBlock = wmma_gemm.GetKPerWaveBlk();
a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`**
```
index_t KInner,
index_t KInner,
a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
make_tuple(I0, m0, k0, I0, I0, I0, I0),
```
