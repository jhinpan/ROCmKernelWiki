# Diff summary

- **files changed:** 31
- **lines:** +856 / -248
- **kernel-ish files:** 29

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+156/-24)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp`  (+60/-40)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`  (+57/-19)
- `example/ck_tile/38_block_scale_gemm/README.md`  (+37/-34)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`  (+39/-21)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb.cpp`  (+59/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshufflequant.cpp`  (+59/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb_preshufflequant.cpp`  (+57/-0)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_flatbr_bquant_cr.hpp`  (+43/-10)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_v2.hpp`  (+43/-8)
- `include/ck_tile/host/tensor_shuffle_utils.hpp`  (+44/-1)
- `example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`  (+22/-16)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped_preshufflequant.cpp`  (+16/-14)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+24/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_quant.cpp`  (+23/-3)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`**
```
typename QuantGroupSize,
GemmConfig::PreshuffleB,
QuantGroupSize>,
```

**`example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`**
```
typename QuantGroupSize,
QuantGroupSize,
typename QuantGroupSize,
const int group_count = arg_parser.get_int("group_count");
```

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp`**
```
using GemmConfig = GemmConfigQuantDecode<T>;
lut[hash_multiple_strings(
{"fp8", "aquant", "non-preshufflequant", "1x1x128"})] = [](const ck_tile::ArgParser&
arg_parser) {
```

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped_preshufflequant.cpp`**
```
using GemmConfig = GemmConfigPreshuffleQuantDecode<T>;
void aquant_quantgrouped_preshufflequant_instance_factory(
{"fp8", "aquant", "preshufflequant", "1x1x128"})] = [](const ck_tile::ArgParser&
arg_parser) {
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8.cpp`**
```
lut[hash_multiple_strings(
{"bf8", "bquant", "non-preshuffleb", "non-preshufflequant", "1x1x64"})] =
lut[hash_multiple_strings(
{"bf8", "bquant", "non-preshuffleb", "non-preshufflequant", "1x1x128"})] =
```
