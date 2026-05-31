# Diff summary

- **files changed:** 38
- **lines:** +2691 / -2076
- **kernel-ish files:** 33

## Files (by churn)

- `csrc/kernels/attention_ragged.cu`  (+55/-1870)
- `csrc/include/attention_common.cuh`  (+1219/-0)
- `op_tests/test_pa_v1.py`  (+604/-0)
- `csrc/kernels/attention_v1.cu`  (+512/-0)
- `csrc/include/mha_fwd.h`  (+50/-73)
- `op_tests/test_mha_varlen.py`  (+45/-24)
- `csrc/include/attention_ragged.h`  (+18/-21)
- `csrc/include/torch/mha_batch_prefill.h`  (+14/-22)
- `csrc/include/torch/mha_varlen_fwd.h`  (+11/-17)
- `aiter/ops/attention.py`  (+22/-1)
- `csrc/include/attention_v1.h`  (+23/-0)
- `csrc/ck_gemm_a8w8/README.md`  (+9/-9)
- `aiter/jit/optCompilerConfig.json`  (+16/-0)
- `aiter/ops/mha.py`  (+15/-0)
- `csrc/ck_batched_gemm_a8w8/README.md`  (+7/-7)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
hipify=True,
hipify=hipify,
hipify = d_args.get("hipify", True)
```

**`aiter/jit/utils/cpp_extension.py`**
```
hipify=True,
if IS_HIP_EXTENSION and with_cuda and hipify:
sources=list(set(sources)),
```

**`aiter/ops/attention.py`**
```
@compile_ops("module_pa_v1")
def paged_attention_v1(
out: torch.Tensor,
workspace_buffer: torch.Tensor,
```

**`aiter/ops/mha.py`**
```
min_seqlen_q: int,
min_seqlen_q: int,
if min_seqlen_q == 0:
md_name += "_nskip"
```

**`csrc/cpp_itfs/mha_fwd_generate.py`**
```
bool use_ext_asm,
bool skip_min_seqlen_q = false)
use_ext_asm,
skip_min_seqlen_q);
```
