# Diff summary

- **files changed:** 22 (diff was byte-capped; summary is partial)
- **lines:** +3312 / -817
- **kernel-ish files:** 8

## Files (by churn)

- `Tensile/ReplacementKernels-cov3/gfx90a/Cijk_Alik_Bljk_SB_MT64x128x32_SE_K1.s.txt`  (+1463/-0)
- `Tensile/ReplacementKernels-cov3/gfx90a/Cijk_Alik_Bljk_SB_MT32x64x32_SE_K1.s.txt`  (+935/-0)
- `Tensile/KernelWriterAssembly.py`  (+316/-339)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT64x128x32_SE_K1.s.txt`  (+203/-204)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT32x64x32_SE_K1.s.txt`  (+68/-71)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT64x32x32_SE_K1.s.txt`  (+68/-71)
- `Tensile/KernelWriter.py`  (+96/-33)
- `Tensile/Common.py`  (+46/-12)
- `Tensile/KernelWriterSource.py`  (+35/-11)
- `Tensile/Components/ShiftVectorComponents.py`  (+12/-15)
- `Tensile/KernelWriterBetaOnly.py`  (+9/-6)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT32x64x32_AF0EM8_ASEM8_FL0_GRVW2_ISA908_MDA2_PGR1_PLR1_SU32_TT2_2_VAW1_VW2_WG16_32_1_WGM8.s.txt`  (+7/-7)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT32x64x32_AF0EM8_ASEM8_FL0_GRVW2_ISA90a_MDA2_PGR1_PLR1_SU32_TT2_2_VAW1_VW2_WG16_32_1_WGM8.s.txt`  (+7/-7)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT64x128x32_AF0EM8_ASEM8_FL0_GRVW4_ISA908_MDA2_PGR1_PLR1_SU32_TT4_4_VAW1_VW4_WG16_32_1_WGM8.s.txt`  (+7/-7)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT64x128x32_AF0EM8_ASEM8_FL0_GRVW4_ISA90a_MDA2_PGR1_PLR1_SU32_TT4_4_VAW1_VW4_WG16_32_1_WGM8.s.txt`  (+7/-7)

## Key added lines (kernel files)

**`Tensile/Common.py`**
```
"PrefetchAcrossPersistentMode": [0, 1],
"MIArchVgpr":               [False, True],
"DepthULdsDivisor":           [1, 2, 4],
{"PrefetchAcrossPersistentMode": [ 0 ] },
```

**`Tensile/Components/ShiftVectorComponents.py`**
```
complexMultiplier = 2 if kernel["ProblemType"]["DataType"].isComplex() else 1
for c  in range(complexMultiplier):
copyInstStr = "v_accvgpr_read_b32" if not kernel["MIArchVgpr"] else "v_mov_b32"
srcVgpr = arch2acc[srcVgpr] * regPerElem + nr + c * accImOffset
```

**`Tensile/Components/Signature.py`**
```
if writer.archCaps["ArchAccUnifiedRegs"]:
agprStart = ceil(totalVgprs/8)*8
vgprCount = agprStart + writer.agprPool.size()
```

**`Tensile/KernelWriter.py`**
```
from .Common import globalParameters, CHeader, roundUp, Backup
from math import ceil
if uDu < kernel["DepthULdsDivisor"] - 1 and kernel.enabledSplitLDS and kernel["PrefetchGlobalRead"]:
if uDu != kernel["DepthULdsDivisor"] - 2 and kernel.enabledSplitLDS:
```

**`Tensile/KernelWriterAssembly.py`**
```
from .DataType import DataType
self.maxOccupancy = 10
def getVgprOccupancy(self, numThreads, vgprs, unifiedVgprRegs=False):
multiplier = int(ceil(max(numThreads, 256) / 256.0)) # example: wg=512 multiplier=2, 1024=4
```
