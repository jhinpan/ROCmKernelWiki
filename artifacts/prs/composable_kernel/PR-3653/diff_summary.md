# Diff summary

- **files changed:** 11
- **lines:** +273 / -208
- **kernel-ish files:** 10

## Files (by churn)

- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+2/-181)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+116/-14)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_splitk_prefill.cpp`  (+64/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_splitk_decode.cpp`  (+61/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+11/-5)
- `test/ck_tile/gemm_block_scale/CMakeLists.txt`  (+11/-0)
- `include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`  (+4/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8.cpp`  (+1/-1)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8i4.cpp`  (+1/-1)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8.cpp`  (+1/-1)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8i4.cpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8.cpp`**
```
using GemmConfig = GemmConfigQuantDecode<T>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8i4.cpp`**
```
using GemmConfig = GemmConfigQuantDecode<T>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8.cpp`**
```
using GemmConfig = GemmConfigQuantDecode<T>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8i4.cpp`**
```
using GemmConfig = GemmConfigQuantDecode<T>;
```

**`include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`**
```
constexpr auto K1 =
GemmPipeline::BlockGemmShape::WarpTile::at(I2); // smallest unit of K work per block
const index_t K_t = amd_wave_read_first_lane(
kargs.k_batch * K1); // amount of K elements consumed if every split-K batch
```
