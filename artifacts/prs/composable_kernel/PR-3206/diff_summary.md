# Diff summary

- **files changed:** 17
- **lines:** +369 / -280
- **kernel-ish files:** 14

## Files (by churn)

- `example/ck_tile/01_fmha/fmha_fwd_runner.hpp`  (+70/-84)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+73/-43)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+55/-49)
- `example/ck_tile/01_fmha/quant.hpp`  (+53/-0)
- `example/ck_tile/01_fmha/script/smoke_test_fwd.sh`  (+15/-29)
- `test/ck_tile/fmha/test_fmha_fwd.cpp`  (+23/-21)
- `example/ck_tile/01_fmha/example_fmha_fwd.cpp`  (+9/-24)
- `include/ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp`  (+31/-0)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+13/-10)
- `example/ck_tile/01_fmha/README.md`  (+6/-13)
- `example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`  (+10/-0)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp`  (+3/-2)
- `include/ck_tile/utility/json_dump.hpp`  (+2/-2)
- `test/ck_tile/fmha/CMakeLists.txt`  (+2/-2)
- `include/ck_tile/host/host_tensor.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
QSCALE_MAP = {
"no": "ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE",
"pertensor": "ck_tile::BlockAttentionQuantScaleEnum::PERTENSOR",
QSCALE_CHECK_MAP = {
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
QSCALE_CHECK_MAP,
QSCALE_MAP,
{F_qscale},
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_dropout}, {F_qscale}, {F_spad}, {F_skpad}, {F_dp
```

**`example/ck_tile/01_fmha/example_fmha_fwd.cpp`**
```
.insert("scale_s", "0", "scale factor of S. 0 means equal to 1/sqrt(hdim)")
.insert("qscale",
"n or 0, no scale\n"
"pt or 1, per-tensor scale\n")
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
const void* q_descale_ptr;
const void* k_descale_ptr;
const void* v_descale_ptr;
args.q_descale_ptr,
```

**`example/ck_tile/01_fmha/fmha_fwd_runner.hpp`**
```
std::string qscale_str,
quant_scale_info qscale = quant_scale_info::decode(qscale_str);
ck_tile::HostTensor<float> q_descale_host(get_lengths(i_perm, 1, 1, 1, 1));
ck_tile::HostTensor<float> k_descale_host(get_lengths(i_perm, 1, 1, 1, 1));
```
