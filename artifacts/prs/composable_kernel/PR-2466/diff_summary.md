# Diff summary

- **files changed:** 55
- **lines:** +1396 / -409
- **kernel-ish files:** 52

## Files (by churn)

- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`  (+106/-87)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp`  (+147/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_kernel_types.hpp`  (+107/-37)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_8bit_traits.hpp`  (+138/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+67/-67)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp`  (+132/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_16bit_traits.hpp`  (+87/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp`  (+86/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp`  (+33/-34)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+49/-14)
- `include/ck_tile/core/arch/generic_memory_space_atomic.hpp`  (+58/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp`  (+24/-25)
- `include/ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp`  (+20/-20)
- `include/ck_tile/ops/gemm/warp/warp_wmma_gemm.hpp`  (+37/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp`  (+14/-14)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
template <typename PrecType>
struct GemmConfigComputeV3_WMMA : public GemmConfigBase
static constexpr ck_tile::index_t M_Tile = 128;
static constexpr ck_tile::index_t N_Tile = 128;
```

**`include/ck_tile/core/arch/arch.hpp`**
```
struct gfx11_t
struct gfx12_t
CK_TILE_DEVICE static constexpr auto get_device_arch()
return gfx11_t{};
```

**`include/ck_tile/core/arch/generic_memory_space_atomic.hpp`**
```
__has_builtin(__builtin_amdgcn_global_atomic_fadd_v2f16) && \
__has_builtin(__builtin_amdgcn_global_atomic_fadd_v2bf16)
CK_TILE_HOST_DEVICE fp16x2_t add_f16x2_t(const fp16x2_t& a, const fp16x2_t& b)
fp16x2_t rtn;
```

**`include/ck_tile/host/device_prop.hpp`**
```
inline bool is_gfx11_supported()
return get_device_name() == "gfx1100" || get_device_name() == "gfx1101" ||
get_device_name() == "gfx1102" || get_device_name() == "gfx1103" ||
get_device_name() == "gfx1150" || get_device_name() == "gfx1151" ||
```

**`include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`**
```
using WG = WarpGemmDispatcher<ATypeToUse,
BTypeToUse,
AccDataType,
isCTransposed>;
```
