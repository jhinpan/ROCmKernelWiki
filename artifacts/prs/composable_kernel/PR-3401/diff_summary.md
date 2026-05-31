# Diff summary

- **files changed:** 9
- **lines:** +231 / -223
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+124/-193)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm.hpp`  (+63/-0)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm.cpp`  (+23/-1)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm_instance.cmake`  (+9/-5)
- `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+4/-9)
- `include/ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp`  (+1/-6)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+3/-4)
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`  (+2/-5)
- `CHANGELOG.md`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/18_flatmm/mxgemm/mx_flatmm.cpp`**
```
.insert("n", "512", "n dimension")
else if(mx_prec == "fp8xfp4")
if(persistent_opt == 0)
return run_mx_flatmm_with_layouts<ck_tile::fp8_t,
```

**`example/ck_tile/18_flatmm/mxgemm/mx_flatmm.hpp`**
```
struct MXf8f4_FlatmmConfig16
static constexpr ck_tile::index_t M_Tile = 128;
static constexpr ck_tile::index_t N_Tile = 256;
static constexpr ck_tile::index_t K_Tile = 256;
```

**`include/ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp`**
```
const auto& c_block_tile = MXFlatmmPipeline{}(a_block_window,
```

**`include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1.hpp`**
```
BDataType_,
static constexpr index_t AK1 = 16 /*dwordx4*/ * APackedSize / sizeof(ADataType);
static constexpr index_t BK1 = 16 /*dwordx4*/ * BPackedSize / sizeof(BDataType);
s_waitcnt_barrier</*vmcnt*/ Bload_num + ScaleAload_num + ScaleBload_num>();
```

**`include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`**
```
namespace detail {
template <typename Problem>
static constexpr index_t DWORDx4            = 16;
using ADataType                      = remove_cvref_t<typename Problem::ADataType>;
```
