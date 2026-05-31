# Diff summary

- **files changed:** 7
- **lines:** +1835 / -584
- **kernel-ish files:** 7

## Files (by churn)

- `kernels/gemm_fp8fp4_gfx1250.py`  (+802/-273)
- `kernels/wmma_gemm_gfx1250.py`  (+478/-191)
- `tests/kernels/test_gemm_fp8fp4_gfx1250.py`  (+364/-31)
- `kernels/gemm_common_gfx1250.py`  (+66/-39)
- `tests/kernels/test_wmma_gemm_gfx1250.py`  (+84/-10)
- `python/flydsl/expr/rocdl/tdm_ops.py`  (+14/-39)
- `kernels/pipeline_utils.py`  (+27/-1)

## Key added lines (kernel files)

**`kernels/gemm_common_gfx1250.py`**
```
from flydsl.expr import arith, buffer_ops, gpu, rocdl, tdm_ops, vector
def _raw_lds_ptr(lds_base_idx, byte_offset):
"""Materialize an LLVM LDS pointer from a pre-extracted byte base."""
return _llvm.inttoptr(lds_ptr_ty, addr_i32)
```

**`kernels/gemm_fp8fp4_gfx1250.py`**
```
from flydsl.expr import arith, buffer_ops, gpu, idx2crd, range_constexpr, rocdl, tdm_ops, vector
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
issue_tdm_loads,
pipeline_fence, pipeline_fence_signal, pipeline_fence_wait,
```

**`kernels/pipeline_utils.py`**
```
def tdm_epilogue_fence_threshold_bytes(*, stage_base_off, tail_plan, loop_iters, extra):
"""Return the earliest stage base that must remain untouched before epilogue.
The TDM-store epilogue reuses the dead LDS prefix starting at byte offset 0.
Reuse is only safe once all stages that may still be consumed after the last
```

**`kernels/wmma_gemm_gfx1250.py`**
```
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
extract_lds_base_idx, get_lds_memref,
issue_tdm_loads,
lds_load_b128_raw, lds_transpose_load_raw,
```

**`python/flydsl/expr/rocdl/tdm_ops.py`**
```
atomic_barrier_enable: bool = False,
lds_byte_offset: Optional extra LDS byte offset applied after the per-wave
LDS address is computed. Use this when multiple descriptors
share the same LDS backing allocation.
```
