# Diff summary

- **files changed:** 28
- **lines:** +675 / -383
- **kernel-ish files:** 14

## Files (by churn)

- `Tensile/Components/MAC_F16_HPA.py`  (+298/-0)
- `Tensile/KernelWriterAssembly.py`  (+10/-255)
- `Tensile/Components/MAC_F16.py`  (+124/-67)
- `Tensile/Tests/unit/test_Priority.py`  (+63/-0)
- `Tensile/Components/Priority.py`  (+58/-0)
- `Tensile/Component.py`  (+34/-21)
- `Tensile/Tests/unit/test_Component.py`  (+42/-8)
- `Tensile/Components/__init__.py`  (+7/-1)
- `Tensile/Perf/DLRM/sgemm_xdlops_nn_terabyte.yaml`  (+4/-4)
- `Tensile/Common.py`  (+5/-1)
- `Tensile/Perf/DLRM/sgemm_xdlops.yaml`  (+3/-3)
- `Tensile/Perf/DLRM/sgemm_xdlops_nt.yaml`  (+3/-3)
- `Tensile/Perf/DLRM/sgemm_xdlops_nt_terabyte.yaml`  (+3/-3)
- `Tensile/Perf/DLRM/sgemm_xdlops_tn_terabyte.yaml`  (+3/-3)
- `Tensile/Perf/TRANSFORMER/sgemm_xdlops.yaml`  (+3/-3)

## Key added lines (kernel files)

**`Tensile/Code.py`**
```
raise NotImplementedError("Half-precision not supported for arch=%u" % self.version )
```

**`Tensile/Common.py`**
```
rv["HasMFMA"]         = tryAssembler(isaVersion, "v_mfma_f32_32x32x2bf16 a[0:31], v32, v33, a[0:31]")
rv["v_mac_f16"]       = tryAssembler(isaVersion, "v_mac_f16 v47, v36, v34")
rv["v_dot2_f32_f16"]  = tryAssembler(isaVersion, "v_dot2_f32_f16 v20, v36, v34, v20")
printExit("Config file requires version=%s is not compatible with current Tensile version=%s" \
```

**`Tensile/Component.py`**
```
if hasattr(pattern, "__call__"):
if not pattern(obj):
if debug:
print("{indent}call({obj}) == False".format(indent=indent, obj=obj))
```

**`Tensile/Components/MAC_F16.py`**
```
from ..Component import Component, MAC
class MAC_Plain(MAC):
Plain MAC instruction implementation
asmCaps = {"v_mac_f16": True,
```

**`Tensile/Components/MAC_F16_HPA.py`**
```
from ..Component import Component, MAC
from ..DataType import DataType
class FMA_HPA_MAD_MIX_LDL(MAC):
def asmCaps(caps):
```
