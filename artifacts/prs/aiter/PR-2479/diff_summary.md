# Diff summary

- **files changed:** 22
- **lines:** +1049 / -545
- **kernel-ish files:** 22

## Files (by churn)

- `csrc/include/moe_sorting_opus.h`  (+313/-304)
- `csrc/kernels/mhc_kernels.cu`  (+308/-7)
- `csrc/include/rocm_ops.hpp`  (+149/-144)
- `op_tests/test_mhc.py`  (+157/-2)
- `csrc/include/aiter_enum.h`  (+25/-23)
- `csrc/include/aiter_hip_common.h`  (+23/-18)
- `csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale_cktile_common.cuh`  (+23/-13)
- `csrc/kernels/mla/metadata/v1_2_device.cuh`  (+10/-9)
- `csrc/include/fused_qk_norm_rope_cache_quant.h`  (+7/-7)
- `csrc/include/activation.h`  (+6/-6)
- `aiter/ops/mhc.py`  (+10/-0)
- `csrc/include/mhc.h`  (+6/-0)
- `csrc/include/opus/opus.hpp`  (+2/-2)
- `op_tests/opus/device/test_opus_device.py`  (+2/-2)
- `aiter/ops/activation.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/mhc.py`**
```
@compile_ops("module_mhc")
def mhc_post(
out: Tensor,
x: Tensor,
```

**`csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale_cktile_common.cuh`**
```
"CKTile blockscale GEMM: XQ inner dim must be contiguous, "
"got strides=[",
XQ.stride(0),
XQ.stride(1),
```

**`csrc/include/activation.h`**
```
void silu_and_mul(torch::Tensor& out, torch::Tensor& input);
void scaled_silu_and_mul(torch::Tensor& out, torch::Tensor& input, torch::Tensor& scale);
void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);
```

**`csrc/include/aiter_enum.h`**
```
typedef enum
switch(dtype)
case AITER_DTYPE_u8: return 1;
case AITER_DTYPE_i16: return 2;
```

**`csrc/include/aiter_hip_common.h`**
```
do                                                                     \
{                                                                      \
if(!(x))                                                           \
{                                                                  \
```
