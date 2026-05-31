# Diff summary

- **files changed:** 14
- **lines:** +84 / -667
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+14/-228)
- `example/ck_tile/01_fmha/fmha_fwd_runner.hpp`  (+50/-180)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+8/-75)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+8/-75)
- `include/ck_tile/host/reference/reference_batched_gemm.hpp`  (+0/-40)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+0/-26)
- `include/ck_tile/core/utility/functional.hpp`  (+0/-12)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+2/-5)
- `example/ck_tile/01_fmha/quant.hpp`  (+0/-7)
- `include/ck_tile/core/numeric/math.hpp`  (+0/-7)
- `include/ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp`  (+0/-6)
- `example/ck_tile/01_fmha/script/smoke_test_fwd.sh`  (+2/-3)
- `example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`  (+0/-2)
- `CHANGELOG.md`  (+0/-1)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
["no", "pertensor"],
["f"], ["no", "pertensor"], get_mask_map(mask_impl).keys(), ["no"]
```

**`example/ck_tile/01_fmha/fmha_fwd_runner.hpp`**
```
auto max_seqlen_k = std::numeric_limits<int32_t>::min();
ck_tile::HostTensor<float> q_descale_host(get_lengths(i_perm, 1, 1, 1, 1));
ck_tile::HostTensor<float> k_descale_host(get_lengths(i_perm, 1, 1, 1, 1));
ck_tile::HostTensor<float> v_descale_host(get_lengths(i_perm, 1, 1, 1, 1));
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`**
```
std::conditional_t<QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR,
FmhaFwdCommonQScaleKargs,
FmhaFwdEmptyKargs<3>>,
std::conditional_t<QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR,
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`**
```
s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
s_acc                  = tile_elementwise_in(s_acc_element_func, s_acc);
s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
auto row_max = scale_s * get_validated_m(m[i_idx]);
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`**
```
s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
s_acc                  = tile_elementwise_in(s_acc_element_func, s_acc);
s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
auto row_max = scale_s * get_validated_m(m[i_idx]);
```
