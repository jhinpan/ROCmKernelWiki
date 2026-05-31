# Diff summary

- **files changed:** 14
- **lines:** +434 / -188
- **kernel-ish files:** 12

## Files (by churn)

- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.cuh`  (+176/-139)
- `aiter/fused_moe.py`  (+117/-5)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_blockscale.cuh`  (+20/-8)
- `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`  (+23/-5)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`  (+18/-6)
- `aiter/ops/moe_op.py`  (+21/-2)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`  (+19/-3)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4_bns.cuh`  (+13/-9)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4.cuh`  (+8/-4)
- `csrc/include/moe_ck.h`  (+6/-2)
- `csrc/include/rocm_ops.hpp`  (+6/-2)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`  (+5/-1)
- `.github/workflows/aiter-test.yaml`  (+1/-1)
- `3rdparty/composable_kernel`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
splitk=0,
get_padded_M(M),  # only used in 2stage
@functools.lru_cache(maxsize=2048)
def get_ksplit(token, topk, expert, inter_dim, model_dim):
```

**`aiter/ops/moe_op.py`**
```
splitk: int = 1,
dst_type: Optional[str] = None,
is_splitk = splitk > 1
outtype = str2dtype_dict[dst_type] if is_splitk else out.dtype
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`**
```
int activation = 0,
int splitk = 1,
std::optional<std::string> dst_type = std::nullopt)
if (splitk > 1)
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`**
```
struct MulABScaleExpertWeightA8W8blkscaleSplitk
template <typename E, typename C, typename D2>
__host__ __device__ constexpr void operator()(E& e, const C& c, const D2& d2) const;
template <>
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.cuh`**
```
template <typename A0DataType,
typename B0DataType,
typename AccDataType,
typename EDataType,
```
