# Diff summary

- **files changed:** 9 (diff was byte-capped; summary is partial)
- **lines:** +3846 / -129
- **kernel-ish files:** 4

## Files (by churn)

- `Tensile/ReplacementKernels-cov3/Cijk_Ailk_Bjlk_DB_MT48x64x4_SE_APM1_AF0EM1_AF1EM1_AMAS3_ASEM1_BL1_DTL0_EPS1_FL1_GRVW2_GSU1_ISA906_IU1_K1_KLA_LPA0_LPB0_LDL1_MI_NLCA1_NLCB1_ONLL1_PBD0_PK0_PGR1_PLR0_RK1_SU0_SNLL0_TT6_4_USFGRO0_VAW1_VW2_WG8_16_1_WGM4.s.txt`  (+2148/-0)
- `Tensile/KernelWriterAssembly.py`  (+494/-108)
- `Tensile/Configs/rocblas_sgemm_tn_inc1_asm_full.yaml`  (+367/-0)
- `Tensile/Configs/rocblas_sgemm_nn_inc1_asm_full.yaml`  (+359/-0)
- `Tensile/Configs/rocblas_sgemm_nt_inc1_asm_full.yaml`  (+335/-0)
- `Tensile/Configs/mfma/mfma_test.yaml`  (+75/-0)
- `Tensile/KernelWriter.py`  (+39/-17)
- `Tensile/Common.py`  (+23/-4)
- `Tensile/KernelWriterSource.py`  (+6/-0)

## Key added lines (kernel files)

**`Tensile/Common.py`**
```
validThreadTileSides = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] + list(range(20, 256, 4))
validMFMA = {}
validMFMA["H"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16]]
validMFMA["S"] = [[32,32,1,2], [32,32,2,1], [16,16,1,4], [16,16,4,1], [4,4,1,16]]
```

**`Tensile/KernelWriter.py`**
```
self.perIterGlobalReadCode = [ Code.Module() for i in range (kernel["LoopIters"]) ]
self.perIterLocalWriteCode = [ Code.Module() for i in range (kernel["LoopIters"]) ]
if endIter > kernel["LoopIters"]-1:
firstStep = endIter-(kernel["LoopIters"]-1) + 1
```

**`Tensile/KernelWriterAssembly.py`**
```
if kernel["MatrixInstruction"] and not self.version == (9,0,8):
printExit("MatrixInstruction not supported for {0}".format(self.version))
if kernel["MatrixInstruction"]:
localReadWidth = tPA["bpe"]//self.bpr # TODO ok for all tile sizes? change for bf16
```

**`Tensile/KernelWriterSource.py`**
```
def MapAcctoArchRegs(self, kernel, option):
return ""
```
