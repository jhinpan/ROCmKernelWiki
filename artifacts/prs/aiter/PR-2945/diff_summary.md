# Diff summary

- **files changed:** 19 (diff was byte-capped; summary is partial)
- **lines:** +2999 / -363
- **kernel-ish files:** 8

## Files (by churn)

- `aiter/ops/opus/README.md`  (+1081/-0)
- `aiter/configs/model_configs/dsv4_bf16_tuned_gemm.csv`  (+350/-350)
- `aiter/configs/model_configs/dsv4_bf16_untuned_gemm.csv`  (+351/-0)
- `aiter/ops/opus/gemm_op_a16w16.py`  (+295/-0)
- `.claude/skills/opus-module-build-optimization/SKILL.md`  (+282/-0)
- `aiter/ops/opus/common.py`  (+208/-0)
- `aiter/ops/opus/_arch.py`  (+136/-0)
- `aiter/ops/opus/__init__.py`  (+86/-0)
- `aiter/jit/utils/cpp_extension.py`  (+56/-3)
- `aiter/ops/deepgemm.py`  (+43/-5)
- `aiter/configs/model_configs/kimi_bf16_untuned_gemm.csv`  (+41/-0)
- `aiter/jit/optCompilerConfig.json`  (+22/-0)
- `aiter/jit/core.py`  (+18/-0)
- `aiter/configs/model_configs/kimi_bf16_tuned_gemm.csv`  (+17/-0)
- `aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv`  (+3/-3)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
from .ops.opus import *  # noqa: F403,E402
```

**`aiter/jit/core.py`**
```
flags_extra_hip_per_source=None,
extra_cuda_cflags_per_source=flags_extra_hip_per_source,
"flags_extra_hip_per_source": {},
if k == "flags_extra_hip_per_source":
```

**`aiter/jit/utils/cpp_extension.py`**
```
extra_cuda_cflags_per_source=None,
extra_cuda_cflags_per_source,
extra_cuda_cflags_per_source=extra_cuda_cflags_per_source,
extra_cuda_cflags_per_source=None,
```

**`aiter/ops/deepgemm.py`**
```
DeepGEMM front-end (CK backend).
Hosts the CK-backed `deepgemm_ck` binding plus a thin `deepgemm()`
wrapper. Opus entries have been extracted under `aiter.ops.opus.*`;
see `aiter.ops.opus.gemm_a16w16_opus` for BF16 matmul and
```

**`aiter/ops/opus/__init__.py`**
```
aiter.ops.opus — opus kernel Python user-facing API.
Per-dtype modules. a16w16 lives here today; a8w8 / a8w8_blockscale
arrive in follow-up PRs. Each module owns its own Python surface and
pybind bindings but shares the underlying JIT module
```
