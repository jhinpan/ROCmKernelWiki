# Diff summary

- **files changed:** 41
- **lines:** +145 / -90
- **kernel-ish files:** 39

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`  (+28/-28)
- `csrc/include/communication_asm.h`  (+27/-14)
- `op_tests/test_mla_persistent.py`  (+28/-5)
- `aiter/mla.py`  (+11/-2)
- `csrc/kernels/mla/metadata/v1_2_device.cuh`  (+6/-4)
- `csrc/py_itfs_cu/asm_mla.cu`  (+8/-0)
- `csrc/kernels/mla/metadata/v1_comm.cuh`  (+3/-4)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/causal_conv1d_split_qkv.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/__init__.py`  (+1/-1)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/__init__.py`  (+1/-1)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_recurrent.py`  (+1/-1)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/gated_delta_rule_utils.py`  (+1/-1)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/__init__.py`  (+1/-1)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/chunk.py`  (+1/-1)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/chunk_delta_h.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/mla.py`**
```
nhead == 16
nhead == 128 and q.dtype == dtypes.fp8 and kv_buffer.dtype == dtypes.fp8
nhead == 32
and q.dtype == dtypes.fp8
```

**`csrc/include/communication_asm.h`**
```
torch::Tensor all_reduce_asm(torch::Tensor& input,
torch::Tensor& reg_sig,
torch::Tensor& reg_buffer,
bool isGraph);
```

**`csrc/kernels/mla/metadata/v1_2_device.cuh`**
```
const hipStream_t stream            = at::hip::getCurrentHIPStream();
(num_heads == 16) || ((num_heads == 32) && q_is_fp8 && kv_is_fp8 && (max_seqlen_qo == 4)) ||
((num_heads == 128) && q_is_fp8 && kv_is_fp8);
TORCH_CHECK((num_heads == 16) || (num_heads == 128) ||
```

**`csrc/py_itfs_cu/asm_mla.cu`**
```
}else if (q_type == "fp8" && kv_type == "fp8"){
if((max_seqlen_q == 4) && persistent){
config_max_seqlen_q = 4;
sub_Q = 128;
```

**`op_tests/test_mla_persistent.py`**
```
if dtype == dtypes.bf16 and nhead == 32:
return False
nhead == 32
and decode_qlen == 4
```
