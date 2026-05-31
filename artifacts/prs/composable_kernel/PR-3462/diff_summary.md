# Diff summary

- **files changed:** 37
- **lines:** +1132 / -677
- **kernel-ish files:** 35

## Files (by churn)

- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshufflequant.cpp`  (+0/-270)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb.cpp`  (+0/-222)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+50/-27)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp`  (+48/-26)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffleQuant_prefill_2d.cpp`  (+63/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb_preshufflequant.cpp`  (+0/-62)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshufflequant_bf8i4.cpp`  (+59/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshufflequant_fp8i4.cpp`  (+59/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb_bf8i4.cpp`  (+57/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb_fp8i4.cpp`  (+57/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshufflequant_bf8.cpp`  (+55/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshufflequant_fp8.cpp`  (+55/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffleQuant_decode_2d.cpp`  (+54/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb_bf8.cpp`  (+53/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb_fp8.cpp`  (+53/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8.cpp`**
```
lut[hash_multiple_strings(
{"bf8", "bquant", "non-preshuffleb", "non-preshufflequant", "1x128x128"})] =
[](const ck_tile::ArgParser& arg_parser) {
using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf8i4.cpp`**
```
lut[hash_multiple_strings(
{"bf8i4", "bquant", "non-preshuffleb", "non-preshufflequant", "1x128x128"})] =
[](const ck_tile::ArgParser& arg_parser) {
using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8.cpp`**
```
lut[hash_multiple_strings(
{"fp8", "bquant", "non-preshuffleb", "non-preshufflequant", "1x128x128"})] =
[](const ck_tile::ArgParser& arg_parser) {
using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8i4.cpp`**
```
lut[hash_multiple_strings(
{"fp8i4", "bquant", "non-preshuffleb", "non-preshufflequant", "1x128x128"})] =
[](const ck_tile::ArgParser& arg_parser) {
using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb_bf8.cpp`**
```
template <typename T>
using GemmConfig = GemmConfigPreshuffleB_BQuant_Prefill_Wmma<T>;
template <typename T>
using GemmConfig = GemmConfigPreshuffleB_BQuant_Prefill<T>;
```
