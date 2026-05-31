# Diff summary

- **files changed:** 15 (diff was byte-capped; summary is partial)
- **lines:** +6739 / -115
- **kernel-ish files:** 14

## Files (by churn)

- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`  (+3009/-0)
- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`  (+1355/-0)
- `python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py`  (+954/-0)
- `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py`  (+585/-0)
- `python/sglang/srt/layers/attention/dsv4/compressor_v2.py`  (+516/-25)
- `python/sglang/srt/layers/attention/dsv4/indexer.py`  (+131/-73)
- `python/sglang/srt/layers/attention/nsa/triton_decode/__init__.py`  (+98/-0)
- `python/sglang/srt/layers/attention/dsv4/compressor.py`  (+37/-1)
- `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`  (+13/-5)
- `python/sglang/srt/layers/attention/hip_flash_mla.py`  (+11/-4)
- `python/sglang/srt/layers/activation.py`  (+14/-0)
- `python/sglang/srt/layers/attention/dsv4/compress_hip.py`  (+5/-4)
- `python/sglang/srt/layers/attention/dsv4/metadata.py`  (+7/-1)
- `python/sglang/srt/environ.py`  (+4/-0)
- `docs/diffusion/compatibility_matrix.md`  (+0/-2)

## Key added lines (kernel files)

**`python/sglang/srt/environ.py`**
```
SGLANG_OPT_USE_AITER_SILU_MUL = EnvBool(False)
SGLANG_OPT_USE_FUSED_COMPRESS_TRITON = EnvBool(False)
SGLANG_OPT_USE_AITER_INDEXER = EnvBool(False)
SGLANG_OPT_USE_JIT_NORM = EnvBool(True)
```

**`python/sglang/srt/layers/activation.py`**
```
get_bool_env_var,
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
from aiter import silu_and_mul as _aiter_silu_and_mul
```

**`python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`**
```
if envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
from sglang.srt.layers.attention.dsv4.compressor_v2 import (
CompressorBackendMixin,
FusedCompressMetadata,
```

**`python/sglang/srt/layers/attention/dsv4/compress_hip.py`**
```
from sglang.srt.layers.attention.dsv4.fused_compress_triton import (
fused_ape_pool_norm_rope,
from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
fused_softmax_pool_triton,
```

**`python/sglang/srt/layers/attention/dsv4/compressor.py`**
```
def _maybe_upgrade_forward_metadata(self) -> None:
if _is_hip:
if not is_paged:
raise NotImplementedError("HIP fused compressor expects paged metadata")
```
