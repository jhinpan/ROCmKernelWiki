# Diff summary

- **files changed:** 29
- **lines:** +929 / -502
- **kernel-ish files:** 16

## Files (by churn)

- `Tensile/SolutionStructs.py`  (+171/-205)
- `Tensile/KernelWriterAssembly.py`  (+273/-96)
- `Tensile/KernelWriter.py`  (+190/-68)
- `Tensile/Components/LraTileAssignment.py`  (+74/-3)
- `Tensile/TensileCreateLibrary.py`  (+31/-31)
- `Tensile/Source/client/source/Reference.cpp`  (+30/-10)
- `Tensile/Source/lib/include/Tensile/MasterSolutionLibrary.hpp`  (+23/-2)
- `Tensile/Source/CMakeLists.txt`  (+7/-8)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT32x64x32_AF0EM8_ASEM8_FL0_GRVW2_ISA908_MDA2_PGR1_PLR1_SU32_TT2_2_VAW1_VW2_WG16_32_1_WGM8.s.txt`  (+7/-7)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT32x64x32_AF0EM8_ASEM8_FL0_GRVW2_ISA90a_MDA2_PGR1_PLR1_SU32_TT2_2_VAW1_VW2_WG16_32_1_WGM8.s.txt`  (+7/-7)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT64x128x32_AF0EM8_ASEM8_FL0_GRVW4_ISA908_MDA2_PGR1_PLR1_SU32_TT4_4_VAW1_VW4_WG16_32_1_WGM8.s.txt`  (+7/-7)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT64x128x32_AF0EM8_ASEM8_FL0_GRVW4_ISA90a_MDA2_PGR1_PLR1_SU32_TT4_4_VAW1_VW4_WG16_32_1_WGM8.s.txt`  (+7/-7)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT64x32x32_AF0EM8_ASEM8_FL0_GRVW2_ISA908_MDA2_PGR1_PLR1_SU32_TT2_2_VAW1_VW2_WG32_16_1_WGM8.s.txt`  (+7/-7)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_SB_MT64x32x32_AF0EM8_ASEM8_FL0_GRVW2_ISA90a_MDA2_PGR1_PLR1_SU32_TT2_2_VAW1_VW2_WG32_16_1_WGM8.s.txt`  (+7/-7)
- `Tensile/ReplacementKernels/Cijk_Alik_Bljk_SB_MT32x64x32_AF0EM8_ASEM8_FL0_GRVW2_ISA908_MDA2_PGR1_PLR1_SU32_TT2_2_VAW1_VW2_WG16_32_1_WGM8.s.txt`  (+7/-7)

## Key added lines (kernel files)

**`Tensile/Common.py`**
```
"DirectToVgprA":              [ False, True ],
"DirectToVgprB":              [ False, True ],
"NonTemporalD":               list(range(0,4)),
{"DirectToVgprA":             [ False ] },
```

**`Tensile/Components/LocalRead.py`**
```
directToLdsStride = False
if kernel["DirectToLds%s" % tP["tensorChar"]] and \
kernel["ProblemType"]["TLU%s" % tP["tensorChar"]] and tP["bpe"] == 8:
directToLdsStride = True
```

**`Tensile/Components/LraTileAssignment.py`**
```
from ..AsmUtils import inst, vgpr, sgpr, log2, vectorStaticDivideAndRemainder, vectorStaticDivide, staticMultiply, vecto
ldsVgpr = writer.vgprPool.checkOut(1,"ldsVgpr")
ldsVgpr1 = writer.vgprPool.checkOut(1,"ldsVgpr1")
if (kernel["DirectToLds%s" % tP["tensorChar"]] and kernel["ProblemType"]["TLU%s" % tP["tensorChar"]] and (kernel["Global
```

**`Tensile/Configuration.py`**
```
elif nodeType == "Constant":
return node.value
assert 0, "Unknown node type: {0}".format(nodeType)
```

**`Tensile/KernelWriter.py`**
```
grBackup = None
doReadA = doReadA and (not kernel["DirectToVgprA"])
doReadB = doReadB and (not kernel["DirectToVgprB"])
readsToWaitAdjust = 0
```
