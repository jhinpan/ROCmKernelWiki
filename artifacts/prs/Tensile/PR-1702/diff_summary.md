# Diff summary

- **files changed:** 17
- **lines:** +2326 / -848
- **kernel-ish files:** 6

## Files (by churn)

- `Tensile/Tests/extended/direct_to_vgpr/dtv_dgemm.yaml`  (+616/-184)
- `Tensile/Tests/extended/direct_to_vgpr/dtv_igemm.yaml`  (+586/-36)
- `Tensile/KernelWriterAssembly.py`  (+214/-253)
- `Tensile/Tests/extended/direct_to_vgpr/dtv_hgemm.yaml`  (+359/-30)
- `Tensile/SolutionStructs.py`  (+145/-125)
- `Tensile/Tests/extended/direct_to_lds/dtl_tsgr_hgemm.yaml`  (+61/-69)
- `Tensile/Tests/extended/direct_to_lds/dtl_hgemm.yaml`  (+58/-66)
- `Tensile/KernelWriter.py`  (+67/-27)
- `Tensile/Tests/extended/direct_to_lds/dtl_tsgr_sgemm.yaml`  (+36/-36)
- `Tensile/Tests/extended/local_split_u/bfloat16_lsu_mfma.yaml`  (+54/-0)
- `Tensile/Tests/extended/local_split_u/sgemm_lsu_mfma.yaml`  (+54/-0)
- `Tensile/Tests/extended/local_split_u/igemm_lsu_mfma.yaml`  (+53/-0)
- `Tensile/Tests/extended/local_split_u/hgemm_lsu_mfma.yaml`  (+16/-15)
- `Tensile/Common.py`  (+3/-3)
- `Tensile/Components/LocalRead.py`  (+2/-2)

## Key added lines (kernel files)

**`Tensile/Common.py`**
```
"AssertFree0ElementMultiple" : [1,2,4,8,16],
"AssertFree1ElementMultiple" : [1,2,4,8,16],
```

**`Tensile/Components/LocalRead.py`**
```
else writer.numVgprValuBPerBlock*writer.numReadsIterCoalescedB*packTimesPerVgpr, "local read pack")
```

**`Tensile/KernelWriter.py`**
```
((kernel["DirectToVgprA"] and (not kernel["ProblemType"]["TLUB"])) or \
(kernel["DirectToVgprB"] and (not kernel["ProblemType"]["TLUA"])))
needExtraLocalReadDo = (NLLlast and isDTVodd and u > localWriteEndIter)
hasLiveLdsData = hasLiveLdsData or needExtraLocalReadDo
```

**`Tensile/KernelWriterAssembly.py`**
```
def closeLoop(self, kernel, loopIdx, finalLoop, loopCopies, uDu=None, emitEndLabelOnly=False, oddLabel=False, skipCondJu
numReadsIterCoalescedA = self.numReadsIterCoalescedA
numReadsIterCoalescedB = self.numReadsIterCoalescedB
numReadsIterCoalesced = max(numReadsIterCoalescedA, numReadsIterCoalescedB)
```

**`Tensile/KernelWriterSource.py`**
```
def closeLoop(self, kernel, loopIdx, finalLoop, loopCopies, uDu=0, emitEndLabelOnly=False, oddLabel=False, skipCondJump=
```
