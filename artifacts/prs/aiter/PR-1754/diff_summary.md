# Diff summary

- **files changed:** 141 (diff was byte-capped; summary is partial)
- **lines:** +2302 / -1691
- **kernel-ish files:** 140

## Files (by churn)

- `csrc/include/custom_all_reduce.cuh`  (+1290/-1296)
- `csrc/py_itfs_ck/mha_batch_prefill_kernels.cu`  (+388/-54)
- `op_tests/test_batch_prefill.py`  (+175/-1)
- `csrc/cpp_itfs/mha_fwd_batch_prefill.cpp`  (+42/-39)
- `csrc/kernels/mla/metadata/v1_0_device.cuh`  (+43/-33)
- `aiter/ops/mha.py`  (+60/-9)
- `csrc/include/rocm_ops.hpp`  (+31/-28)
- `csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages_common.cuh`  (+25/-31)
- `csrc/include/mha_fwd.h`  (+36/-0)
- `csrc/include/groupnorm.hpp`  (+18/-15)
- `csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages.h`  (+22/-9)
- `csrc/include/custom_all_reduce.h`  (+10/-10)
- `csrc/include/aiter_hip_common.h`  (+9/-10)
- `csrc/ck_deepgemm/include/deepgemm_common.cuh`  (+5/-6)
- `csrc/cktile_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle_cktile_common.cuh`  (+5/-6)

## Key added lines (kernel files)

**`aiter/dist/device_communicators/communicator_cuda.py`**
```
and (input_.nelement() * input_.element_size())
```

**`aiter/jit/core.py`**
```
except Exception:
```

**`aiter/ops/mha.py`**
```
kv_last_page_lens: Optional[Tensor] = None,
block_table: Optional[Tensor] = None,
seqlen_k: Optional[Tensor] = None,
bias: Optional[torch.Tensor] = None,
```

**`aiter/ops/triton/__init__.py`**
```
These following help implement backward-compatibility
```

**`aiter/utility/dtypes.py`**
```
- "512" -> 512 (single value without comma returns int)
- "512," -> (512,) (trailing comma returns tuple)
- "512,1024" -> (512, 1024) (multiple values return tuple)
```
