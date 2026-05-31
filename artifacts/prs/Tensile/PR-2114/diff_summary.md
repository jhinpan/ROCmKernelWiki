# Diff summary

- **files changed:** 29
- **lines:** +150 / -32
- **kernel-ish files:** 8

## Files (by churn)

- `Tensile/Components/MFMA.py`  (+50/-0)
- `Tensile/AsmCaps.py`  (+44/-0)
- `Tensile/Common.py`  (+10/-8)
- `Tensile/SolutionStructs.py`  (+10/-3)
- `Tensile/Source/lib/include/Tensile/AMDGPU.hpp`  (+7/-0)
- `Tensile/KernelWriterAssembly.py`  (+3/-1)
- `Tensile/Source/lib/include/Tensile/PlaceholderLibrary.hpp`  (+3/-0)
- `Tensile/Tests/emulation/mfma/hpa_bfloat16_gemm_asm.yaml`  (+1/-1)
- `Tensile/Tests/extended/custom_kernel/ck_dgemm_90a_nn.yaml`  (+1/-1)
- `Tensile/Tests/extended/custom_kernel/ck_dgemm_90a_nn_large_offset.yaml`  (+1/-1)
- `Tensile/Tests/extended/direct_to_lds/dtl_tsgr_f8.yaml`  (+1/-1)
- `Tensile/Tests/extended/direct_to_vgpr/dtv_igemm.yaml`  (+1/-1)
- `Tensile/Tests/extended/local_split_u/igemm_lsu_mfma.yaml`  (+1/-1)
- `Tensile/Tests/pre_checkin/denorm/mfma/bfloat16_denorm.yaml`  (+1/-1)
- `Tensile/Tests/pre_checkin/denorm/mfma/hgemm_denorm_alt.yaml`  (+1/-1)

## Key added lines (kernel files)

**`Tensile/AsmCaps.py`**
```
(9, 5, 0): {'HasAddLshl': True,
'HasAtomicAdd': True,
'HasDirectToLdsDest': False,
'HasDirectToLdsNoDest': True,
```

**`Tensile/Common.py`**
```
(9,4,2), (9,5,0),
'gfx950':'gfx950', 'gfx950:xnack+':'gfx950', 'gfx950:xnack-':'gfx950',
validMFMA["H"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [16,16,32,1], [4,4,4,16]]
isaVersion==(9,4,2) or isaVersion==(9,5,0))
```

**`Tensile/Components/MFMA.py`**
```
class MFMASelection950(MFMA):
versions = [(9,5,0)]
def WaitCount(self, writer):
kernel = writer.kernel
```

**`Tensile/KernelWriterAssembly.py`**
```
'b64'       : 'dwordx2',
'b128'      : 'dwordx4',
def longBranch(self, label, tmpSgpr=None):
```

**`Tensile/SolutionStructs.py`**
```
isa = tuple(state["ISA"])
if not globalParameters["ArchCaps"][isa]["HasDTLx4"]:
if numBytesPerLoad != 4:
reject(state, "DirectToLds can only be used with buffer loads requiring 1 register")
```
