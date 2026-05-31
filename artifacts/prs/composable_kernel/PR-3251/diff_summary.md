# Diff summary

- **files changed:** 37
- **lines:** +1044 / -1868
- **kernel-ish files:** 37

## Files (by churn)

- `test/ck_tile/grouped_gemm_preshuffle/test_grouped_gemm_preshuffle_util.hpp`  (+81/-142)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`  (+79/-131)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_preshuffle.cpp`  (+66/-102)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+66/-98)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+62/-94)
- `example/ck_tile/22_gemm_multi_abd/gemm_multi_abd_fp16.cpp`  (+62/-93)
- `example/ck_tile/19_gemm_multi_d/gemm_multi_d_fp16.cpp`  (+60/-91)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_util.hpp`  (+39/-73)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+37/-66)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_multi_d.cpp`  (+29/-66)
- `test/ck_tile/grouped_gemm_multi_d/test_grouped_gemm_multi_d_util.hpp`  (+23/-57)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle_invoker.hpp`  (+22/-55)
- `test/ck_tile/gemm_multi_abd/test_gemm_multi_abd_util.hpp`  (+24/-52)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage_invoker.hpp`  (+21/-54)
- `example/ck_tile/03_gemm/universal_gemm_invoker.hpp`  (+22/-53)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_invoker.hpp`**
```
constexpr auto scheduler = GemmConfig::Scheduler;
using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
BDataType,
AccDataType,
```

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`**
```
constexpr auto scheduler = GemmConfig::Scheduler;
const auto Run = [&]() {
constexpr auto memory_operation = ck_tile::memory_operation_enum::set;
scheduler>;
```

**`example/ck_tile/03_gemm/gemm_weight_preshuffle_invoker.hpp`**
```
constexpr auto scheduler = GemmConfig::Scheduler;
using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
BDataType,
AccDataType,
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
using UniversalGemmProblem =
ck_tile::UniversalGemmPipelineProblem<ADataType,
BDataType,
AccDataType,
```

**`example/ck_tile/03_gemm/universal_gemm_invoker.hpp`**
```
constexpr auto scheduler = GemmConfig::Scheduler;
using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
BDataType,
AccDataType,
```
