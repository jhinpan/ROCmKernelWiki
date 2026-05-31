# Diff summary

- **files changed:** 36
- **lines:** +1391 / -0
- **kernel-ish files:** 32

## Files (by churn)

- `test/ck_tile/batched_transpose/batched_transpose.inc`  (+283/-0)
- `test/ck_tile/smoothquant/smoothquant.inc`  (+274/-0)
- `test/ck_tile/smoothquant/instances/smoothquant_fwd_api.cpp`  (+143/-0)
- `test/ck_tile/smoothquant/smoothquant.hpp`  (+114/-0)
- `test/ck_tile/batched_transpose/batched_transpose_api.cpp`  (+113/-0)
- `test/ck_tile/smoothquant/instances/smoothquant_instance_common.hpp`  (+61/-0)
- `test/ck_tile/batched_transpose/CMakeLists.txt`  (+33/-0)
- `test/ck_tile/smoothquant/CMakeLists.txt`  (+28/-0)
- `test/ck_tile/batched_transpose/batched_transpose.hpp`  (+25/-0)
- `test/ck_tile/smoothquant/instances/smoothquant_bf16_n1024_instance.cpp`  (+21/-0)
- `test/ck_tile/smoothquant/instances/smoothquant_fp16_n1024_instance.cpp`  (+21/-0)
- `test/ck_tile/smoothquant/instances/smoothquant_bf16_n2048_instance.cpp`  (+13/-0)
- `test/ck_tile/smoothquant/instances/smoothquant_bf16_n3072_instance.cpp`  (+13/-0)
- `test/ck_tile/smoothquant/instances/smoothquant_bf16_n4096_instance.cpp`  (+13/-0)
- `test/ck_tile/smoothquant/instances/smoothquant_bf16_n4096_tp_instance.cpp`  (+13/-0)

## Key added lines (kernel files)

**`test/ck_tile/batched_transpose/batched_transpose.hpp`**
```
struct batched_transpose_trait
std::string type;
std::string layout;
struct batched_transpose_kargs : public ck_tile::BatchedTransposeHostArgs
```

**`test/ck_tile/batched_transpose/batched_transpose.inc`**
```
template <typename DataType>
auto get_elimit(std::string /*init_method*/)
double rtol = 1e-3;
double atol = 1e-3;
```

**`test/ck_tile/batched_transpose/batched_transpose_api.cpp`**
```
template <typename ts_type,
ck_tile::index_t block_x,
ck_tile::index_t block_y,
ck_tile::index_t warp_x,
```

**`test/ck_tile/batched_transpose/batched_transpose_bf16.cpp`**
```
int main()
std::vector<std::vector<std::string>> test_cases = generate_test_cases("bf16");
return !run_test_cases<ck_tile::bf16_t>(test_cases);
```

**`test/ck_tile/batched_transpose/batched_transpose_fp16.cpp`**
```
int main()
std::vector<std::vector<std::string>> test_cases = generate_test_cases("fp16");
return !run_test_cases<ck_tile::fp16_t>(test_cases);
```
