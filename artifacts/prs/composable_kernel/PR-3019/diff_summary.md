# Diff summary

- **files changed:** 8
- **lines:** +157 / -94
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck_tile/host/shuffle_utils.hpp`  (+75/-0)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+24/-48)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+38/-31)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`  (+1/-13)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+7/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+6/-1)
- `example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`  (+5/-1)
- `include/ck_tile/host.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`**
```
ck_tile::memory_operation_enum::set,
GemmConfig::TiledMMAPermuteN>>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
static constexpr bool TiledMMAPermuteN = false;
static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
static constexpr bool TiledMMAPermuteN = N_Repeat % 2 == 0;
static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
ck_tile::shuffle_aq(aq_tensor_ptr.get(), GemmConfig::K_Tile / QuantGroupSize);
if constexpr(GemmConfig::PreshuffleB)
if constexpr(GemmConfig::TiledMMAPermuteN)
printf("PreshuffleB with TiledMMAPermuteN\n");
```

**`include/ck_tile/host/shuffle_utils.hpp`**
```
namespace ck_tile {
template <typename T>
auto shuffle_aq(const ck_tile::HostTensor<T>* t, int block_aq_k)
if(t->get_lengths().size() != 2)
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`**
```
static constexpr bool TiledMMAPermuteN        = GemmConfig::TiledMMAPermuteN;
```
