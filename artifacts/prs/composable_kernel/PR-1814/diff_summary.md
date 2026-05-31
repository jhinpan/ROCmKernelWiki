# Diff summary

- **files changed:** 9
- **lines:** +67 / -19
- **kernel-ish files:** 5

## Files (by churn)

- `example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`  (+28/-4)
- `example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.cpp`  (+19/-3)
- `example/ck_tile/02_layernorm2d/generate.py`  (+5/-3)
- `example/ck_tile/10_rmsnorm2d/generate.py`  (+5/-3)
- `include/ck_tile/host/check_err.hpp`  (+5/-1)
- `example/ck_tile/10_rmsnorm2d/script/smoke_test.sh`  (+2/-2)
- `example/ck_tile/02_layernorm2d/CMakeLists.txt`  (+1/-1)
- `example/ck_tile/02_layernorm2d/script/smoke_test.sh`  (+1/-1)
- `example/ck_tile/10_rmsnorm2d/CMakeLists.txt`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/02_layernorm2d/generate.py`**
```
'int8' : 'ck_tile::int8_t',
'fp8'  : 'ck_tile::fp8_t'}
dynamic_quant_out_dtype = ['int8', 'fp8']
('fp16,int8'), ('bf16,int8'),
```

**`example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`**
```
template <>
auto get_elimit<ck_tile::int8_t>()
double rtol = 1e-2;
double atol = 1.0;
```

**`example/ck_tile/10_rmsnorm2d/generate.py`**
```
'int8' : 'ck_tile::int8_t',
'fp8'  : 'ck_tile::fp8_t'}
dynamic_quant_out_dtype = ['int8', 'fp8']
('fp16,int8'), ('bf16,int8'),
```

**`example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.cpp`**
```
if((fused_quant == 1 || fused_quant == 2) && prec_o != "int8" && prec_o != "fp8")
std::cout
<< "if fused_quant is 1 or 2, only support \"-prec_o=int8\" or \"-prec_o=fp8\" cases."
<< std::endl;
```

**`include/ck_tile/host/check_err.hpp`**
```
const float error_percent =
static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
std::cerr << "max err: " << max_err;
std::cerr << ", number of errors: " << err_count;
```
