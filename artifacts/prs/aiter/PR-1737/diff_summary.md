# Diff summary

- **files changed:** 12
- **lines:** +683 / -370
- **kernel-ish files:** 11

## Files (by churn)

- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages_common.py`  (+229/-170)
- `csrc/ck_tile_gemm_moe_2stages/gen_instances.py`  (+155/-109)
- `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages.cu`  (+188/-67)
- `aiter/fused_moe.py`  (+62/-14)
- `csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages.h`  (+21/-3)
- `aiter/ops/moe_op.py`  (+12/-0)
- `csrc/include/rocm_ops.hpp`  (+6/-2)
- `csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages_common.cuh`  (+4/-1)
- `op_tests/test_moe_2stage.py`  (+3/-1)
- `3rdparty/composable_kernel`  (+1/-1)
- `csrc/cpp_itfs/gluon_aot_tools/extra/hip/compile.h`  (+1/-1)
- `csrc/kernels/mla/metadata/v1_0_device.cuh`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
tgN = (inter_dim + tileN - 1) // tileN
if n <= 1:
if q_type in [QuantType.per_1x128, QuantType.per_1x32]
activation=activation,
```

**`aiter/ops/moe_op.py`**
```
activation: Optional[int] = 0,
split_k: Optional[int] = 1,
activation: Optional[int] = 0,
split_k: Optional[int] = 1,
```

**`csrc/ck_tile_gemm_moe_2stages/gen_instances.py`**
```
import itertools
act_dict,
dtype_dict,
is_split_k,
```

**`csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages.h`**
```
std::optional<torch::Tensor>,
std::optional<int>,
std::optional<int>)>;
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType, int activation, bool kHasBia
```

**`csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages_common.cuh`**
```
int ActivationOp,
std::conditional_t<ActivationOp == 2, ck_tile::moe::Swiglu, ck_tile::moe::MoeSilu>;
```
