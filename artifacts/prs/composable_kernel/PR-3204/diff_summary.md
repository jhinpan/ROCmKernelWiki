# Diff summary

- **files changed:** 9
- **lines:** +31 / -36
- **kernel-ish files:** 9

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+10/-11)
- `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`  (+5/-6)
- `include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`  (+3/-5)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm_tile_partitioner.hpp`  (+3/-4)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+3/-3)
- `include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`  (+2/-2)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp`  (+2/-2)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+2/-2)
- `include/ck_tile/ops/flatmm/kernel/moe_flatmm_kernel.hpp`  (+1/-1)

## Key added lines (kernel files)

**`include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`**
```
constexpr auto N1   = BlockGemmShape::WarpTile::at(number<1>{});
constexpr auto K1   = BlockGemmShape::WarpTile::at(number<2>{});
```

**`include/ck_tile/ops/flatmm/kernel/moe_flatmm_kernel.hpp`**
```
constexpr auto K1   = BlockGemmShape::WarpTile::at(number<2>{});
```

**`include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`**
```
static constexpr uint32_t MPerBlock = BlockGemmShapeType::kM;
static constexpr uint32_t NPerBlock = BlockGemmShapeType::kN;
static constexpr uint32_t KPerBlock = BlockGemmShapeType::kK;
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_tile_partitioner.hpp`**
```
static constexpr index_t MPerBlock                          = BlockGemmShapeType::kM;
static constexpr index_t NPerBlock                          = BlockGemmShapeType::kN;
static constexpr index_t KPerBlock                          = BlockGemmShapeType::kK;
```

**`include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`**
```
constexpr auto K1   = GemmPipeline::BlockGemmShape::WarpTile::at(number<2>{});
static_assert(!GemmPipeline::BlockGemmShape::PermuteA, "Not implemented!");
if constexpr(GemmPipeline::BlockGemmShape::PermuteB)
if constexpr(GemmPipeline::BlockGemmShape::PermuteB)
```
