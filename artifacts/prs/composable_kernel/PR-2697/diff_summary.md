# Diff summary

- **files changed:** 21
- **lines:** +1245 / -782
- **kernel-ish files:** 20

## Files (by churn)

- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+14/-259)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage_invoker.hpp`  (+259/-0)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+33/-190)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle_invoker.hpp`  (+204/-0)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`  (+6/-192)
- `example/ck_tile/03_gemm/universal_gemm_invoker.hpp`  (+197/-0)
- `example/ck_tile/03_gemm/gemm_basic_invoker.hpp`  (+176/-0)
- `example/ck_tile/21_elementwise/elementwise_example_unary.cpp`  (+72/-18)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+27/-38)
- `example/ck_tile/03_gemm/run_gemm_example_common.hpp`  (+64/-0)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`  (+27/-29)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage.cpp`  (+52/-0)
- `example/ck_tile/21_elementwise/elementwise_example.cpp`  (+26/-18)
- `example/ck_tile/21_elementwise/elementwise_example_add_4d.cpp`  (+23/-14)
- `example/ck_tile/21_elementwise/elementwise_example_transpose.cpp`  (+17/-10)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
using GemmConfig = GemmConfigBase;
using Invoker    = BasicInvoker;
return run_gemm_example_prec_type<GemmConfig, Invoker, ck_tile::half_t>(
a_layout, b_layout, arg_parser);
```

**`example/ck_tile/03_gemm/gemm_basic_invoker.hpp`**
```
struct BasicInvoker
template <typename GemmConfig,
typename ADataType,
typename BDataType,
```

**`example/ck_tile/03_gemm/gemm_splitk_two_stage.cpp`**
```
int run_gemm_example(ck_tile::ArgParser& arg_parser)
std::string data_type = arg_parser.get_str("prec");
std::string a_layout  = arg_parser.get_str("a_layout");
std::string b_layout  = arg_parser.get_str("b_layout");
```

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_invoker.hpp`**
```
template <typename PrecType_, typename WorkspaceType_>
struct GemmConfigTwoStage : public GemmConfigComputeV3<PrecType_>
using WorkspaceType = ck_tile::remove_cvref_t<WorkspaceType_>;
struct SplitKTwoStageInvoker
```

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`**
```
int run_gemm_example_with_layouts_two_stage(ck_tile::ArgParser& arg_parser,
int run_gemm_example_prec_type(std::string a_layout,
std::string b_layout,
ck_tile::ArgParser& arg_parser)
```
