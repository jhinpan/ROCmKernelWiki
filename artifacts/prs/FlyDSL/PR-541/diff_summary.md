# Diff summary

- **files changed:** 8
- **lines:** +134 / -267
- **kernel-ish files:** 8

## Files (by churn)

- `kernels/rmsnorm_kernel.py`  (+59/-101)
- `kernels/fp8_gemm_4wave.py`  (+19/-48)
- `kernels/fp8_gemm_8wave.py`  (+19/-48)
- `kernels/layernorm_kernel.py`  (+14/-26)
- `kernels/softmax_kernel.py`  (+9/-22)
- `kernels/fp8_gemm_utils.py`  (+13/-9)
- `kernels/topk_gating_softmax_kernel.py`  (+0/-12)
- `python/flydsl/expr/primitive.py`  (+1/-1)

## Key added lines (kernel files)

**`kernels/fp8_gemm_4wave.py`**
```
@fx.struct
class SharedStorage:
A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
```

**`kernels/fp8_gemm_8wave.py`**
```
@fx.struct
class SharedStorage:
A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
```

**`kernels/fp8_gemm_utils.py`**
```
step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
sum_i32 = base_i32 + fx.Int32(step_off)
lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
```

**`kernels/layernorm_kernel.py`**
```
@fx.struct
class SharedStorage:
s_sum: fx.Array[fx.Float32, RED_SLOTS, 16]
s_sumsq: fx.Array[fx.Float32, RED_SLOTS, 16]
```

**`kernels/rmsnorm_kernel.py`**
```
def _make_reduction_storage(red_slots: int):
@fx.struct
class SharedStorage:
s_red: fx.Array[fx.Float32, red_slots, 16]
```
