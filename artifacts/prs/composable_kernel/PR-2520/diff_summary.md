# Diff summary

- **files changed:** 28
- **lines:** +2116 / -1541
- **kernel-ish files:** 28

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`  (+1169/-0)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+59/-958)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+116/-115)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_run_test.inc`  (+113/-112)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`  (+105/-104)
- `include/ck_tile/ops/gemm/kernel/gemm_multi_d_kernel.hpp`  (+185/-0)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+103/-69)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+128/-37)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+14/-16)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_util.hpp`  (+15/-15)
- `test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_util.hpp`  (+14/-16)
- `example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`  (+14/-15)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+14/-15)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_util.hpp`  (+14/-14)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+13/-13)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
```

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s);
```

**`example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`**
```
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
const auto Run = [&](const auto has_hot_loop_,
const auto tail_number_,
const auto memory_operation_) {
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s);
ck_tile::GemmHostArgs args = {a_m_k_dev_buf.GetDeviceBuffer(),
b_k_n_dev_buf.GetDeviceBuffer(),
c_m_n_dev_buf.GetDeviceBuffer(),
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
const auto Run = [&](const auto has_hot_loop_,
const auto tail_number_,
const auto memory_operation_) {
```
