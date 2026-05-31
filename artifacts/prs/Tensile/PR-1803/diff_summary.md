# Diff summary

- **files changed:** 20
- **lines:** +357 / -170
- **kernel-ish files:** 4

## Files (by churn)

- `Tensile/SolutionStructs.py`  (+184/-100)
- `Tensile/KernelWriterAssembly.py`  (+80/-37)
- `Tensile/Common.py`  (+20/-6)
- `Tensile/KernelWriter.py`  (+19/-4)
- `Tensile/Tests/pre_checkin/mfma/wider_local_read.yaml`  (+12/-6)
- `Tensile/Tests/emulation/mfma/1LDSB.yaml`  (+6/-3)
- `Tensile/Tests/pre_checkin/mfma/1LDSB.yaml`  (+6/-3)
- `Tensile/Tests/extended/direct_to_vgpr/dtv_hgemm.yaml`  (+4/-2)
- `Tensile/Tests/extended/custom_kernel/ck_dgemm_90a_nn.yaml`  (+4/-0)
- `Tensile/Tests/extended/custom_kernel/ck_dgemm_90a_nn_large_offset.yaml`  (+4/-0)
- `Tensile/Tests/extended/local_split_u/sgemm_lsu_mfma.yaml`  (+2/-2)
- `Tensile/Tests/extended/direct_to_vgpr/dtv_igemm.yaml`  (+2/-1)
- `Tensile/Tests/pre_checkin/mfma/dgemm_gb_global_ldd.yaml`  (+2/-1)
- `Tensile/Tests/pre_checkin/wmma/hgemm_wmma.yaml`  (+2/-1)
- `Tensile/Tests/pre_checkin/wmma/hpa_bfloat16_gemm_wmma.yaml`  (+2/-1)

## Key added lines (kernel files)

**`Tensile/Common.py`**
```
"LdsPadA":                     list(range(-1, 128)),
"LdsPadB":                     list(range(-1, 128)),
"LdsBlockSizePerPadA":          [-1, 0, 64, 128, 256, 512, 1024, 2048, 4096],
"LdsBlockSizePerPadB":          [-1, 0, 64, 128, 256, 512, 1024, 2048, 4096],
```

**`Tensile/KernelWriter.py`**
```
latencyForLR += kernel["ExtraLatencyForLR"]
if self.miLatency <= 4 and kernel["LoopIters"] >= 4:
numMfmaBetweenLWandBarrier *= 2
if packItems and self.miLatencyLeft > 2:
```

**`Tensile/KernelWriterAssembly.py`**
```
kStrSLDS = ""
if tailLoop and kernel.enabledSplitLDS:
tailLoopLabelEnd = self.getNamedLabel(
"TailLoopEnd%s%s"%(loopChar, "_G2L%s"%(kernel["DepthULdsDivisor"]-1) if kernel.enabledSplitLDS else "") )
```

**`Tensile/SolutionStructs.py`**
```
@staticmethod
def getLdsNumElements(state, tc):
bpeAB = int(4*state["ProblemType"]["DataType"].numRegisters())
if state["LdsBlockSizePerPad%s"%tc]:
```
