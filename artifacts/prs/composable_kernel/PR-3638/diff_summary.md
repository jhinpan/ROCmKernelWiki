# Diff summary

- **files changed:** 32
- **lines:** +182 / -213
- **kernel-ish files:** 31

## Files (by churn)

- `example/ck_tile/38_block_scale_gemm/gemm_quant.cpp`  (+2/-68)
- `include/ck_tile/host/tensor_shuffle_utils.hpp`  (+23/-19)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp`  (+27/-15)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_abquant_pipeline_ag_bg_cr_base_policy.hpp`  (+16/-4)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+7/-9)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+5/-5)
- `example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`  (+4/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp`  (+4/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped_preshufflequant.cpp`  (+4/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf16mxfp4.cpp`  (+4/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8.cpp`  (+4/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8i4.cpp`  (+4/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8.cpp`  (+4/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8i4.cpp`  (+4/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb_bf8.cpp`  (+4/-4)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`**
```
static auto _ = []() {
auto& lut                               = get_kernel_lut();
return 0;
```

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp`**
```
static auto _ = []() {
auto& lut            = get_kernel_lut();
return 0;
```

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped_preshufflequant.cpp`**
```
static auto _ = []() {
auto& lut            = get_kernel_lut();
return 0;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf16mxfp4.cpp`**
```
static auto _ = []() {
auto& lut        = get_kernel_lut();
return 0;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8.cpp`**
```
static auto _ = []() {
auto& lut = get_kernel_lut();
return 0;
```
