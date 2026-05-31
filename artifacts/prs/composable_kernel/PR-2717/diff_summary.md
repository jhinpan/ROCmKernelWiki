# Diff summary

- **files changed:** 11
- **lines:** +112 / -222
- **kernel-ish files:** 11

## Files (by churn)

- `test/ck_tile/gemm/test_gemm_pipeline_universal_run_test.inc`  (+56/-96)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_run_test.inc`  (+47/-84)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_bf16.cpp`  (+1/-8)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_bf8.cpp`  (+1/-8)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_fp16.cpp`  (+1/-8)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_fp8.cpp`  (+1/-8)
- `test/ck_tile/gemm/test_gemm_pipeline_smoke_run_test.inc`  (+1/-6)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_bf16.cpp`  (+1/-1)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_bf8.cpp`  (+1/-1)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_fp16.cpp`  (+1/-1)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_fp8.cpp`  (+1/-1)

## Key added lines (kernel files)

**`test/ck_tile/gemm/test_gemm_pipeline_basic_bf16.cpp`**
```
int main() { return run_gemm_combinations<ck_tile::bf16_t>(); }
```

**`test/ck_tile/gemm/test_gemm_pipeline_basic_bf8.cpp`**
```
int main() { return run_gemm_combinations<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>(); }
```

**`test/ck_tile/gemm/test_gemm_pipeline_basic_fp16.cpp`**
```
int main() { return run_gemm_combinations<ck_tile::half_t>(); }
```

**`test/ck_tile/gemm/test_gemm_pipeline_basic_fp8.cpp`**
```
int main() { return run_gemm_combinations<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>(); }
```

**`test/ck_tile/gemm/test_gemm_pipeline_basic_run_test.inc`**
```
bool run_gemm_test_prec_type(std::string a_layout,
std::string b_layout,
ck_tile::ArgParser& arg_parser)
arg_parser, Row{}, Col{}, Row{});
```
