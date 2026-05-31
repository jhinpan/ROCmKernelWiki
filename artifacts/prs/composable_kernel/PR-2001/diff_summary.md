# Diff summary

- **files changed:** 10
- **lines:** +485 / -352
- **kernel-ish files:** 10

## Files (by churn)

- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+104/-77)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+103/-76)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+60/-38)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+57/-40)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+58/-30)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+32/-31)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+14/-39)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+28/-6)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_util.hpp`  (+28/-6)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+1/-9)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
const auto Run = [&](const auto memory_operation_) {
constexpr auto memory_operation = memory_operation_.value;
using GemmEpilogue = ck_tile::CShuffleEpilogue<
ck_tile::CShuffleEpilogueProblem<ADataType,
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
const auto Run = [&](const auto has_hot_loop_,
const auto tail_number_,
const auto memory_operation_) {
constexpr bool has_hot_loop_v   = has_hot_loop_.value;
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
const auto Run =
[&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
constexpr bool has_hot_loop_v   = has_hot_loop_.value;
constexpr auto tail_number_v    = tail_number_.value;
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
const auto Run =
[&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
constexpr bool has_hot_loop_v   = has_hot_loop_.value;
constexpr auto tail_number_v    = tail_number_.value;
```

**`include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`**
```
bool isCTransposed_,
memory_operation_enum MemoryOperation_>
using ADataType                                        = remove_cvref_t<ADataType_>;
using BDataType                                        = remove_cvref_t<BDataType_>;
```
