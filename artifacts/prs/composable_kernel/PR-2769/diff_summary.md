# Diff summary

- **files changed:** 16
- **lines:** +198 / -55
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`  (+72/-1)
- `include/ck_tile/core/numeric/bfloat16.hpp`  (+11/-23)
- `test/ck_tile/gemm/test_gemm_pipeline_smoke_util.hpp`  (+27/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_int8.cpp`  (+7/-8)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_pk_int4.cpp`  (+7/-8)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_fp16.cpp`  (+10/-1)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_bf8.cpp`  (+9/-1)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_fp8.cpp`  (+9/-1)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_bf8.cpp`  (+9/-1)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_fp8.cpp`  (+9/-1)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_bf16.cpp`  (+8/-1)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_bf16.cpp`  (+8/-1)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_fp16.cpp`  (+8/-1)
- `test/ck_tile/elementwise/CMakeLists.txt`  (+1/-4)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_run_test.inc`  (+2/-2)

## Key added lines (kernel files)

**`include/ck_tile/core/numeric/bfloat16.hpp`**
```
uint32_t bits = bit_cast<uint32_t>(f);
if(~bits & 0x7f800000)
bits += 0x7fff + ((bits >> 16) & 1); // Round to nearest, round to even
else if(bits & 0xffff)
```

**`include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`**
```
template <typename T, std::size_t N, typename F, std::size_t... Is>
constexpr std::array<T, N> make_lookup_table_impl(F&& func, std::index_sequence<Is...>)
return {func(Is)...};
template <typename T, std::size_t N, typename F>
```

**`test/ck_tile/gemm/test_gemm_pipeline_basic_bf16.cpp`**
```
int main()
bool is_success = true;
is_success      = run_gemm_combinations<ck_tile::bf16_t>() && is_success;
is_success =
```

**`test/ck_tile/gemm/test_gemm_pipeline_basic_bf8.cpp`**
```
int main()
bool is_success = true;
is_success =
run_gemm_combinations<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>() && is_success;
```

**`test/ck_tile/gemm/test_gemm_pipeline_basic_fp16.cpp`**
```
int main()
bool is_success = true;
is_success      = run_gemm_combinations<ck_tile::half_t>() && is_success;
is_success =
```
