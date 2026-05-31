# Diff summary

- **files changed:** 8
- **lines:** +103 / -29
- **kernel-ish files:** 8

## Files (by churn)

- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`  (+41/-2)
- `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`  (+18/-7)
- `aiter/fused_moe.py`  (+12/-5)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`  (+9/-7)
- `aiter/ops/moe_op.py`  (+8/-2)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`  (+7/-2)
- `csrc/include/moe_ck.h`  (+4/-2)
- `csrc/include/rocm_ops.hpp`  (+4/-2)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
isShuffled = getattr(w1, "is_shuffled", False)
isShuffled,
is_shuffled=True,
and is_shuffled
```

**`aiter/ops/moe_op.py`**
```
is_shuffled: bool = True,
is_shuffled,
is_shuffled: bool = True,
is_shuffled,
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`**
```
MoeKernel moe_dispatch(std::string &kernelName, int block_m, int inter_dim, at::ScalarType x_dtype, at::ScalarType w_dty
return moe_stage1_heuristic_dispatch(block_m, inter_dim, x_dtype, w_dtype, y_dtype, act_op, quant_type, mul_routed_weigh
return moe_stage2_heuristic_dispatch(block_m, inter_dim, x_dtype, w_dtype, y_dtype, 0, quant_type, mul_routed_weight, is
std::optional<std::string> dst_type = std::nullopt,
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`**
```
struct MulABScaleShuffled
template <typename E, typename C, typename D0, typename D1, typename D2>
__host__ __device__ constexpr void
operator()(E& e, const C& c, const D0& d0, const D1& d1, const D2& d2) const;
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`**
```
elif tag == "a8w8" or tag == "a4w4_bns":
elif tag == "a4w4":
kernel.CDEElementOp = "MulABScaleShuffled"
elif tag == "a8w8" or tag == "a4w4_bns":
```
