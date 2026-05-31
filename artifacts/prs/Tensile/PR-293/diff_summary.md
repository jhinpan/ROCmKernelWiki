# Diff summary

- **files changed:** 22
- **lines:** +1230 / -247
- **kernel-ish files:** 11

## Files (by churn)

- `Tensile/KernelWriterAssembly.py`  (+305/-125)
- `Tensile/Tests/pre_checkin/test_hgemm_hpa_asm_nn.yaml`  (+132/-0)
- `Tensile/Tests/bugs/hgemm_lsu.yaml`  (+0/-92)
- `Tensile/Tests/nightly/global_split_u/hgemm_gsu.yaml`  (+80/-0)
- `Tensile/Tests/nightly/local_split_u/hgemm_lsu_grvw1.yaml`  (+80/-0)
- `Tensile/Tests/nightly/local_split_u/sgemm_lsu.yaml`  (+80/-0)
- `Tensile/Tests/nightly/local_split_u/hgemm_lsu.yaml`  (+79/-0)
- `Tensile/Tests/nightly/global_split_u/sgemm_gsu.yaml`  (+78/-0)
- `Tensile/Tests/nightly/local_split_u/dgemm_lsu.yaml`  (+77/-0)
- `Tensile/Tests/pre_checkin/test_hgemm_hpa_asm_tn.yaml`  (+71/-0)
- `Tensile/Tests/pre_checkin/test_hgemm_hpa_asm_nt.yaml`  (+69/-0)
- `Tensile/Tests/pre_checkin/test_hgemm_hpa_asm_tt.yaml`  (+66/-0)
- `Tensile/Source/Client.h`  (+31/-12)
- `Tensile/SolutionStructs.py`  (+30/-4)
- `Tensile/Tests/nightly/local_split_u/test_local_split_u.py`  (+14/-0)

## Key added lines (kernel files)

**`Tensile/ClientWriter.py`**
```
h += "const unsigned printTensorA=%x;\n" % int(globalParameters["PrintTensorA"])
h += "const unsigned printTensorB=%x;\n" % int(globalParameters["PrintTensorB"])
h += "const unsigned printTensorC=%x;\n" % int(globalParameters["PrintTensorC"])
h += "    if (strideA != std::numeric_limits<unsigned int>::max())  strideA%u%s = strideA;\n" % (lastStrideA-1, indexCha
```

**`Tensile/Common.py`**
```
globalParameters["PrintTensorA"] = 0          # Print TensorA after initialization
globalParameters["PrintTensorB"] = 0          # Print TensorB after initialization
globalParameters["PrintTensorC"] = 0          # Print TensorC.  0x1=after init; 0x2=after copy-back; 0x3=both
"VectorAtomicWidth":          [ -1, 1, 2 ] ,
```

**`Tensile/KernelWriterAssembly.py`**
```
self.do["ApplyAlpha"]  = True
def getLocalSplitUElementStep(self, kernel, isLds):
if isLds and \
kernel["VectorWidth"]*self.bpeCinternal >= 8 and \
```

**`Tensile/SolutionStructs.py`**
```
supported = \
state["ProblemType"]["DataType"].isSingle() or \
(state["KernelLanguage"] == "Assembly" and \
(state["ProblemType"]["DataType"].isHalf() or \
```

**`Tensile/Source/Client.h`**
```
if (printTensorC & 0x1) {
printTensor("C_in", initialC, numIndicesC[problemTypeIdx],
if (printTensorC & 0x2) {
printTensor("C_result", deviceOnHostC, numIndicesC[problemTypeIdx],
```
