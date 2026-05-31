# Diff summary

- **files changed:** 49 (diff was byte-capped; summary is partial)
- **lines:** +4033 / -865
- **kernel-ish files:** 33

## Files (by churn)

- `include/rapidjson/document.h`  (+1091/-0)
- `example/ck_tile/01_fmha/fmha_bwd.cpp`  (+382/-324)
- `example/include/json_dump.hpp`  (+700/-0)
- `include/rapidjson/allocators.h`  (+693/-0)
- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+265/-223)
- `example/ck_tile/19_gemm_multi_d/run_gemm_multi_d_fp16_example.inc`  (+66/-47)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+64/-42)
- `example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`  (+61/-40)
- `example/ck_tile/18_flatmm/run_flatmm_example.inc`  (+50/-33)
- `include/rapidjson/cursorstreamwrapper.h`  (+78/-0)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+39/-30)
- `example/ck_tile/15_fused_moe/main.cpp`  (+50/-1)
- `example/ck_tile/13_moe_sorting/README.md`  (+26/-22)
- `example/ck_tile/16_batched_gemm/README.md`  (+21/-19)
- `example/ck_tile/15_fused_moe/README.md`  (+38/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/fmha_bwd.cpp`**
```
"will not be used")
.insert("json", "0", "0: No Json, 1: Dump Results in Json format")
.insert("jsonfile", "fmha_bwd.json", "json file name to dump results");
bool pass = true;
```

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
.insert("repeat", "20", "number of iterations to benchmark the kernel")
.insert("json", "0", "0: No Json, 1: Dump Results in Json format")
.insert("jsonfile", "fmha_fwd.json", "json file name to dump results");
bool pass = true;
```

**`example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`**
```
.insert("repeat", "20", "hot iter")
.insert("json", "0", "0: No Json, 1: Dump Results in Json format")
.insert("jsonfile", "layernorm2d_fwd.json", "json file name to dump results");
if(arg_parser.get_int("json") == 1)
```

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
.insert("json", "0", "0: No Json, 1: Dump Results in Json format")
.insert("jsonfile", "gemm.json", "json file name to dump results")
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
float ave_time = invoke_gemm<GemmConfig,
ADataType,
BDataType,
ck_tile::tuple<>,
```
