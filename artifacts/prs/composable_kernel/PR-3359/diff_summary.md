# Diff summary

- **files changed:** 42 (diff was byte-capped; summary is partial)
- **lines:** +1809 / -2306
- **kernel-ish files:** 42

## Files (by churn)

- `include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`  (+241/-221)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_two_stage_invoker.hpp`  (+133/-153)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage_invoker.hpp`  (+131/-147)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_multi_d.cpp`  (+93/-131)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+88/-124)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_preshuffle.cpp`  (+85/-124)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`  (+91/-100)
- `example/ck_tile/03_gemm/universal_gemm_invoker.hpp`  (+81/-99)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle_invoker.hpp`  (+78/-97)
- `example/ck_tile/03_gemm/gemm_basic_invoker.hpp`  (+75/-94)
- `example/ck_tile/40_streamk_gemm/streamk_gemm_basic.cpp`  (+71/-96)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight_invoker.hpp`  (+71/-90)
- `example/ck_tile/17_grouped_gemm/quant_invoke_grouped_gemm_kernel.hpp`  (+73/-86)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data_invoker.hpp`  (+68/-82)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`  (+64/-81)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic_invoker.hpp`**
```
using GemmEpilogue = ck_tile::CShuffleEpilogue<
ck_tile::CShuffleEpilogueProblem<ADataType,
BDataType,
ck_tile::tuple<>,
```

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_invoker.hpp`**
```
using GemmEpilogue = ck_tile::CShuffleEpilogue<
ck_tile::CShuffleEpilogueProblem<ADataType,
BDataType,
DsDataType,
```

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`**
```
using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
BDataType,
AccDataType,
GemmShape,
```

**`example/ck_tile/03_gemm/gemm_weight_preshuffle_invoker.hpp`**
```
using GemmEpilogue = ck_tile::CShuffleEpilogue<
ck_tile::CShuffleEpilogueProblem<ADataType,
BDataType,
DsDataType,
```

**`example/ck_tile/03_gemm/universal_gemm_invoker.hpp`**
```
using GemmEpilogue = ck_tile::CShuffleEpilogue<
ck_tile::CShuffleEpilogueProblem<ADataType,
BDataType,
DsDataType,
```
