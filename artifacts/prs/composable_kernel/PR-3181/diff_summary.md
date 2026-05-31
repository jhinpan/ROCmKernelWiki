# Diff summary

- **files changed:** 14
- **lines:** +805 / -495
- **kernel-ish files:** 12

## Files (by churn)

- `example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`  (+0/-428)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+266/-7)
- `example/ck_tile/38_block_scale_gemm/gemm_quant.cpp`  (+130/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+13/-41)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp`  (+53/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffleb_prefill.cpp`  (+53/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_prefill_bf8i4.cpp`  (+49/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_prefill_fp8i4.cpp`  (+49/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_prefill_bf8.cpp`  (+47/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_prefill_fp8.cpp`  (+47/-0)
- `example/ck_tile/38_block_scale_gemm/README.md`  (+25/-17)
- `example/ck_tile/38_block_scale_gemm/gemm_quant_rowcol.cpp`  (+30/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_quant_tensor.cpp`  (+30/-0)
- `example/ck_tile/38_block_scale_gemm/CMakeLists.txt`  (+13/-2)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp`**
```
template <typename T>
using GemmConfig = GemmConfigQuant<T>;
void aquant_quantgrouped_instance_factory(
std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut)
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_prefill_bf8.cpp`**
```
template <typename T>
using GemmConfig = GemmConfigBQuantPrefill<T>;
run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>, \
TypeConfig,                 \
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_prefill_bf8i4.cpp`**
```
template <typename T>
using GemmConfig = GemmConfigBQuantPrefill<T>;
run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>, \
TypeConfig,                 \
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_prefill_fp8.cpp`**
```
template <typename T>
using GemmConfig = GemmConfigBQuantPrefill<T>;
run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>, \
TypeConfig,                 \
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_prefill_fp8i4.cpp`**
```
template <typename T>
using GemmConfig = GemmConfigBQuantPrefill<T>;
run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>, \
TypeConfig,                 \
```
