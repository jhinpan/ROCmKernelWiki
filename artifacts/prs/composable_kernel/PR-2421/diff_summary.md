# Diff summary

- **files changed:** 51 (diff was byte-capped; summary is partial)
- **lines:** +5146 / -0
- **kernel-ish files:** 46

## Files (by churn)

- `test/ck_tile/layernorm2d/generate.py`  (+730/-0)
- `test/ck_tile/rmsnorm2d/generate.py`  (+689/-0)
- `test/ck_tile/layernorm2d/layernorm2d_fwd.inc`  (+566/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_smoke_run_test.inc`  (+458/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_smoke_util.hpp`  (+414/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_run_test.inc`  (+393/-0)
- `test/ck_tile/add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.inc`  (+370/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_run_test.inc`  (+313/-0)
- `test/ck_tile/add_rmsnorm2d_rdquant/instances/add_rmsnorm2d_rdquant_fwd_api.cpp`  (+227/-0)
- `test/ck_tile/add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`  (+151/-0)
- `test/ck_tile/add_rmsnorm2d_rdquant/instances/add_rmsnorm2d_rdquant_fwd_instance_common.hpp`  (+70/-0)
- `test/ck_tile/layernorm2d/layernorm2d_fwd.hpp`  (+70/-0)
- `test/ck_tile/rmsnorm2d/CMakeLists.txt`  (+54/-0)
- `test/ck_tile/layernorm2d/CMakeLists.txt`  (+53/-0)
- `test/ck_tile/add_rmsnorm2d_rdquant/instances/add_rmsnorm2d_rdquant_fwd_bf16_n8192_instance.cpp`  (+42/-0)

## Key added lines (kernel files)

**`test/ck_tile/add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`**
```
template <typename InputDataType, typename QuantizedDataType>
struct AddRmsnormRdquantTypeConfig;
template <>
struct AddRmsnormRdquantTypeConfig<ck_tile::half_t, ck_tile::int8_t>
```

**`test/ck_tile/add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.inc`**
```
template <typename InputDataType>
auto get_elimit()
double rtol = 1e-2;
double atol = 1e-2;
```

**`test/ck_tile/add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd_bf16.cpp`**
```
int main() { return run_add_rmsnorm2d_rdquant_combinations("bf16"); }
```

**`test/ck_tile/add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd_fp16.cpp`**
```
int main() { return run_add_rmsnorm2d_rdquant_combinations("fp16"); }
```

**`test/ck_tile/add_rmsnorm2d_rdquant/instances/add_rmsnorm2d_rdquant_fwd_api.cpp`**
```
template <typename InputDataType_,
typename QuantizedDataType_,
ck_tile::index_t Repeat_M_,         // each thread repeat along M
ck_tile::index_t Repeat_N_,         // each thread repeat along N
```
