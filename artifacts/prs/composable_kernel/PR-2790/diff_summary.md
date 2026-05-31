# Diff summary

- **files changed:** 14
- **lines:** +224 / -67
- **kernel-ish files:** 14

## Files (by churn)

- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+61/-10)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+37/-10)
- `example/ck_tile/18_flatmm/run_flatmm_example.inc`  (+34/-9)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp`  (+18/-9)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+16/-8)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp`  (+14/-10)
- `example/ck_tile/18_flatmm/flatmm_basic.hpp`  (+10/-0)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+5/-5)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v1.hpp`  (+5/-5)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+8/-0)
- `include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`  (+4/-1)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`  (+4/-0)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_preshuffle.cpp`  (+4/-0)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+4/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
template <typename PrecType>
struct GemmConfigPreshufflePrefill_Wmma : public GemmConfigPreshufflePrefill<PrecType>
static constexpr ck_tile::index_t M_Warp_Tile = 16;
static constexpr ck_tile::index_t N_Warp_Tile = 16;
```

**`example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`**
```
return !run_gemm_example<GemmConfigPreshufflePrefill_Wmma>(arg_parser);
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
int n_ = t.get_lengths()[1];
int k_ = t.get_lengths()[0];
if(ck_tile::is_gfx12_supported())
constexpr int divisor      = 2;
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`**
```
template <typename PrecType>
struct GemmConfigPreshuffleDecode_Wmma : public GemmConfigBase
static constexpr ck_tile::index_t M_Tile = 32 / sizeof(PrecType);
static constexpr ck_tile::index_t N_Tile = 64;
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm_preshuffle.cpp`**
```
return !run_grouped_gemm_example<GemmConfigPreshuffleDecode_Wmma>(argc, argv);
```
