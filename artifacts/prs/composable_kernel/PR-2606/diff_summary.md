# Diff summary

- **files changed:** 11
- **lines:** +683 / -127
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+346/-40)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp`  (+121/-40)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp`  (+139/-16)
- `include/ck/host_utility/device_prop.hpp`  (+21/-16)
- `include/ck/utility/get_id.hpp`  (+34/-1)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp`  (+5/-4)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp`  (+5/-3)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_b_scale.hpp`  (+5/-3)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_ab_scale.hpp`  (+3/-2)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp`  (+3/-1)
- `include/ck/utility/blkgemmpipe_scheduler.hpp`  (+1/-1)

## Key added lines (kernel files)

**`include/ck/host_utility/device_prop.hpp`**
```
inline bool is_gfx12_supported()
return ck::get_device_name() == "gfx1200" || ck::get_device_name() == "gfx1201";
inline bool is_gfx11_supported()
return ck::get_device_name() == "gfx1100" || ck::get_device_name() == "gfx1101" ||
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp`**
```
static constexpr index_t MWaves   = MPerBlock / (MRepeat * MPerXDL);
static constexpr index_t NWaves   = NPerBlock / (NRepeat * NPerXDL);
static constexpr index_t WaveSize = BlockSize / MWaves / NWaves;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp`**
```
using Base::WaveSize;
(4 * WaveSize / BlockSize) >= 1 ? 4 * WaveSize / BlockSize : 1;
using Base::WaveSize;
(4 * WaveSize / BlockSize) >= 1 ? 4 * WaveSize / BlockSize : 1;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_ab_scale.hpp`**
```
using Base::WaveSize;
(4 * WaveSize / BlockSize) >= 1 ? 4 * WaveSize / BlockSize : 1;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_b_scale.hpp`**
```
using Base::WaveSize;
(4 * WaveSize / BlockSize) >= 1 ? 4 * WaveSize / BlockSize : 1;
using Base::WaveSize;
(4 * WaveSize / BlockSize) >= 1 ? 4 * WaveSize / BlockSize : 1;
```
