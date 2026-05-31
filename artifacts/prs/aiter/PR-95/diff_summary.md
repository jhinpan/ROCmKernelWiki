# Diff summary

- **files changed:** 35
- **lines:** +1628 / -192
- **kernel-ish files:** 32

## Files (by churn)

- `op_tests/test_moe_2stage.py`  (+365/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm_common.cuh`  (+297/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_kernel.cu`  (+243/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm.hpp`  (+216/-0)
- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+64/-64)
- `aiter/fused_moe_bf16_asm.py`  (+92/-16)
- `csrc/include/moe_op.h`  (+49/-49)
- `aiter/ops/moe_op.py`  (+41/-8)
- `csrc/pybind/moe_ck_2stages_pybind.cu`  (+33/-0)
- `csrc/py_itfs_ck/moe_sorting_kernels.cu`  (+15/-16)
- `csrc/include/moe_ck.h`  (+26/-1)
- `aiter/ops/quant.py`  (+15/-8)
- `csrc/include/aiter_hip_common.h`  (+9/-9)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_b16_f8.cu`  (+16/-0)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_f16_f8.cu`  (+16/-0)

## Key added lines (kernel files)

**`aiter/fused_moe_bf16_asm.py`**
```
def moe_sorting_ck(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, block_size=BLOCK_SIZE_M, expert_mask=No
num_valid_ids = torch.empty((1 + num_experts + 1),
dtype=torch.int32,
device=device)
```

**`aiter/jit/core.py`**
```
import multiprocessing
if multiprocessing.current_process().name == 'MainProcess':
shutil.copytree(CK_DIR, f'{bd_dir}/ck', dirs_exist_ok=True)
if os.path.exists(f'{bd_dir}/ck/library'):
```

**`aiter/ops/moe_op.py`**
```
num_valid_ids: Tensor,
num_valid_ids: Tensor,
num_valid_ids: Tensor,
num_valid_ids: Tensor,
```

**`aiter/ops/quant.py`**
```
hidden_states = x * x_scale
per_token_scale = per_token_amax / dtypeMax
x = x.to(torch.float)
scale = torch.abs(x).max() / dtypeMax
```

**`csrc/include/aiter_hip_common.h`**
```
do                                                                                                                      
{                                                                                                                       
hipError_t err = call;                                                                                                  
if (err != hipSuccess)                                                                                                  
```
