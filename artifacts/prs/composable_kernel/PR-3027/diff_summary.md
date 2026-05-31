# Diff summary

- **files changed:** 15
- **lines:** +1513 / -720
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+184/-658)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_global.hpp`  (+405/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_thread_tiles.hpp`  (+402/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_wave_tiles.hpp`  (+343/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`  (+97/-36)
- `include/ck/utility/amd_transpose_load.hpp`  (+37/-0)
- `example/01_gemm/gemm_wmma_fp16_v3.cpp`  (+9/-8)
- `include/ck/utility/synchronization.hpp`  (+10/-6)
- `include/ck/utility/dynamic_buffer.hpp`  (+12/-1)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+6/-6)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_b_scale.hpp`  (+4/-5)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_f16_f16_f16/device_gemm_wmma_universal_f16_f16_f16_km_kn_mn.hpp`  (+1/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_f16_f16_f16/device_gemm_wmma_universal_f16_f16_f16_km_nk_mn.hpp`  (+1/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_f16_f16_f16/device_gemm_wmma_universal_f16_f16_f16_mk_kn_mn.hpp`  (+1/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_f16_f16_f16/device_gemm_wmma_universal_f16_f16_f16_mk_nk_mn.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_wmma_fp16_v3.cpp`**
```
128, 256, 64,
S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>,
S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>,
S<1, 64, 1, 4>, 8,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`**
```
using Base::I1;
using Base::WaveSize;
using typename Base::HotLoopInstList;
make_tuple(I0, I0, I0, I0, I0, I0),
```

**`include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_global.hpp`**
```
namespace ck {
template <typename SrcDesc,
typename DstDesc,
typename SrcData,
```

**`include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_thread_tiles.hpp`**
```
namespace ck {
template <typename ABLayout,
typename ABMajorLayout,
typename LDSTypeAB,
```

**`include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_wave_tiles.hpp`**
```
namespace ck {
template <typename ABLayout,
typename ABMajorLayout,
typename LDSTypeAB,
```
