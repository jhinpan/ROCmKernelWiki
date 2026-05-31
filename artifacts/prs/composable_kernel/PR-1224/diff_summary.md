# Diff summary

- **files changed:** 21
- **lines:** +709 / -261
- **kernel-ish files:** 18

## Files (by churn)

- `include/ck_tile/core/numeric/math.hpp`  (+186/-13)
- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+114/-26)
- `example/ck_tile/01_fmha/generate.py`  (+71/-58)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+57/-50)
- `include/ck_tile/core/utility/unary_element_function.hpp`  (+67/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+30/-15)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+30/-15)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qs_ks_vs.hpp`  (+30/-15)
- `include/ck_tile/host/check_err.hpp`  (+28/-9)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+15/-12)
- `example/ck_tile/01_fmha/README.md`  (+16/-10)
- `example/ck_tile/01_fmha/script/benchmark.sh`  (+17/-6)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_fp8.hpp`  (+11/-11)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp`  (+9/-7)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp`  (+8/-7)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
.insert("scale_s",
"scale factor of S. 0 means equal to 1/sqrt(hdim).\n"
"note when squant=1, this value will be modified by range_q/k")
.insert("range_q", "16", "per-tensor quantization range of q. used if squant=1.")
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
using BiasDataType        = float;
float scale_s;
float scale_p;
float scale_o;
```

**`example/ck_tile/01_fmha/generate.py`**
```
{F_squant},
using fmha_kernel_{F_idx} =
{F_pipeline_enum}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_squant}, {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}>;
FMHA_FWD_API_INNER_DISPATCH="""            {F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && (
```

**`include/ck_tile/core/numeric/math.hpp`**
```
template <typename Scale, Scale lhs>
struct scales_c
template <typename Right>
CK_TILE_HOST_DEVICE constexpr auto operator()(const Right& rhs) const -> decltype(lhs * rhs)
```

**`include/ck_tile/core/utility/unary_element_function.hpp`**
```
namespace ck_tile {
template <typename F, typename... Fs>
struct composes : private composes<F>
template <typename FirstArg, typename... RestArgs>
```
