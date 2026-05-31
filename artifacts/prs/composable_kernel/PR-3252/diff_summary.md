# Diff summary

- **files changed:** 202
- **lines:** +576 / -591
- **kernel-ish files:** 202

## Files (by churn)

- `client_example/12_elementwise_normalization/elementwise_layernorm2d.cpp`  (+176/-176)
- `client_example/23_elementwise_transpose/elementwise_transpose_3d.cpp`  (+140/-140)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n1024_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n1536_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n2048_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n256_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n3072_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n4096_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n4096_tp_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n512_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n64_n128_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_bf16_n768_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_fp16_n1024_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_fp16_n1536_instance.cpp`  (+2/-3)
- `test/ck_tile/moe_smoothquant/instances/moe_smoothquant_fp16_n2048_instance.cpp`  (+2/-3)

## Key added lines (kernel files)

**`client_example/12_elementwise_normalization/elementwise_layernorm2d.cpp`**
```
using ADataType             = ck::half_t; // Input 1
using BDataType             = ck::half_t; // Input 2
using XDataType             = ck::half_t;
using GammaDataType         = ck::half_t;
```

**`client_example/23_elementwise_transpose/elementwise_transpose_3d.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using ADataType = F16;
using BDataType = F16;
```
