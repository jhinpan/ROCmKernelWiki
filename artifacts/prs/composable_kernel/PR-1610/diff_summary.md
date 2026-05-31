# Diff summary

- **files changed:** 13
- **lines:** +114 / -38
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+94/-18)
- `include/ck_tile/ops/reduce/block/block_reduce.hpp`  (+9/-9)
- `include/ck_tile/core/numeric/math.hpp`  (+1/-1)
- `include/ck_tile/host/reference/reference_elementwise.hpp`  (+1/-1)
- `include/ck_tile/host/reference/reference_permute.hpp`  (+1/-1)
- `include/ck_tile/host/reference/reference_rmsnorm2d_fwd.hpp`  (+1/-1)
- `include/ck_tile/ops/add_rmsnorm2d_rdquant/kernel/add_rmsnorm2d_rdquant_fwd_shape.hpp`  (+1/-1)
- `include/ck_tile/ops/add_rmsnorm2d_rdquant/pipeline/add_rmsnorm2d_rdquant_fwd_pipeline_problem.hpp`  (+1/-1)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp`  (+1/-1)
- `include/ck_tile/ops/permute/pipeline/generic_petmute_problem.hpp`  (+1/-1)
- `include/ck_tile/ops/rmsnorm2d/kernel/rmsnorm2d_fwd_shape.hpp`  (+1/-1)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_problem.hpp`  (+1/-1)
- `include/ck_tile/ops/welford/block/block_welford.hpp`  (+1/-1)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`**
```
__host__ __device__ ~UnaryOpBase() = default;
__host__ __device__ constexpr UnaryOpBase()                   = default;
__host__ __device__ constexpr UnaryOpBase(const UnaryOpBase&) = default;
__host__ __device__ constexpr UnaryOpBase(UnaryOpBase&&)      = default;
```

**`include/ck_tile/ops/reduce/block/block_reduce.hpp`**
```
const ReduceFunc& reduce_func,
bool_constant<WithBroadcast> = {})
const ReduceFunc& reduce_func)
const InDistributedTensor_& in_tensor,
```
