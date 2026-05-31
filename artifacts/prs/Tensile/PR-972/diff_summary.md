# Diff summary

- **files changed:** 12
- **lines:** +2507 / -1711
- **kernel-ish files:** 7

## Files (by churn)

- `Tensile/KernelWriterAssembly.py`  (+1069/-1167)
- `Tensile/Tests/pre_checkin/mfma/hpa_bfloat16_gemm_asm.yaml`  (+434/-0)
- `Tensile/Tests/pre_checkin/mfma/hpa_hgemm_asm.yaml`  (+430/-0)
- `Tensile/KernelWriter.py`  (+175/-199)
- `Tensile/SolutionStructs.py`  (+212/-119)
- `Tensile/Tests/pre_checkin/mfma/sgemm.yaml`  (+163/-39)
- `Tensile/Tests/pre_checkin/mfma/hpa_bfloat16_gemm_asm_mi32x32x2x2.yaml`  (+0/-91)
- `Tensile/Tests/pre_checkin/mfma/hpa_hgemm_asm_mi32x32x4x2.yaml`  (+0/-87)
- `Tensile/KernelWriterSource.py`  (+8/-6)
- `Tensile/DataType.py`  (+10/-0)
- `Tensile/Common.py`  (+4/-2)
- `Tensile/Code.py`  (+2/-1)

## Key added lines (kernel files)

**`Tensile/Code.py`**
```
self.name     = name
self.tempVgpr = None
```

**`Tensile/Common.py`**
```
"LdsBlockSizePerPad":          [-1, 0, 64, 128, 256],
{"LdsBlockSizePerPad":        [ 0 ] },
```

**`Tensile/DataType.py`**
```
'miInput' : 1,
'miInput' : 1,
'miInput' : 1,
'miInput' : 1,
```

**`Tensile/KernelWriter.py`**
```
self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
numMfmaPerIter = kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1] * kernel["InnerUnroll"]
if kernel["EnableMatrixInstruction"]:
```

**`Tensile/KernelWriterAssembly.py`**
```
if kernel["EnableMatrixInstruction"] and not self.version == (9,0,8):
localReadWidth = (kernel["VectorWidth"] * tPA["bpe"]) // self.bpr
if kernel["EnableMatrixInstruction"]:
localReadWidth = tPA["bpe"] / self.bpr
```
