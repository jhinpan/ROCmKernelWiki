# Diff summary

- **files changed:** 12
- **lines:** +627 / -127
- **kernel-ish files:** 11

## Files (by churn)

- `aiter/ops/triton/fused_mxfp4_quant.py`  (+194/-1)
- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages_common.py`  (+128/-32)
- `csrc/ck_tile_gemm_moe_2stages/gen_instances.py`  (+107/-45)
- `csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages_common.cuh`  (+43/-27)
- `aiter/fused_moe.py`  (+44/-5)
- `op_tests/test_moe_2stage.py`  (+37/-12)
- `aiter/ops/quant.py`  (+37/-0)
- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages.cu`  (+33/-1)
- `3rdparty/composable_kernel`  (+1/-1)
- `aiter/ops/triton/utils/gemm_config_utils.py`  (+1/-1)
- `csrc/include/mha_fwd.h`  (+1/-1)
- `hsa/gfx942/fmha_v3_fwd/codegen.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
bf16_fp8_bound = 512
if quant_type == QuantType.per_1x32 and M < bf16_fp8_bound:
q_dtype_a = dtypes.bf16
elif quant_type == QuantType.per_1x32 and M >= bf16_fp8_bound:
```

**`aiter/ops/quant.py`**
```
def per_1x32_f8_scale_f8_quant(
x, scale=None, quant_dtype=dtypes.fp8, scale_type=dtypes.fp32, shuffle=False
assert quant_dtype == dtypes.fp8
block_size = 32
```

**`aiter/ops/triton/fused_mxfp4_quant.py`**
```
@triton.jit
def _fused_quant_fp8_sort_kernel(
input_ptr,
sorted_ids_ptr,
```

**`csrc/ck_tile_gemm_moe_2stages/gen_instances.py`**
```
a_dtypes,
self.init = True
self.dispatchers_path = os.path.join(working_path, "dispatchers")
self.kernel_name_list = []
```

**`csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages_common.cuh`**
```
constexpr bool AQUANT_Pipeline = std::is_same_v<ADataType, ck_tile::bf8_t> ||
std::is_same_v<ADataType, ck_tile::fp8_t> ||
std::is_same_v<ADataType, ck_tile::pk_fp4_t>;
constexpr bool BMXFP4_Pipeline = std::is_same_v<BDataType, ck_tile::pk_fp4_t>;
```
