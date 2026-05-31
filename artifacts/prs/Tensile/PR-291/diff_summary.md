# Diff summary

- **files changed:** 18
- **lines:** +895 / -171
- **kernel-ish files:** 9

## Files (by churn)

- `Tensile/KernelWriterAssembly.py`  (+174/-57)
- `Tensile/Tests/pre_checkin/test_hgemm_hpa_asm_nn.yaml`  (+132/-0)
- `Tensile/Tests/bugs/hgemm_lsu.yaml`  (+0/-92)
- `Tensile/Tests/nightly/local_split_u/test_hgemm_lsu_grvw1.yaml`  (+80/-0)
- `Tensile/Tests/nightly/local_split_u/test_sgemm_lsu.yaml`  (+80/-0)
- `Tensile/Tests/nightly/local_split_u/test_hgemm_lsu.yaml`  (+79/-0)
- `Tensile/Tests/nightly/local_split_u/test_dgemm_lsu.yaml`  (+77/-0)
- `Tensile/Tests/pre_checkin/test_hgemm_hpa_asm_tn.yaml`  (+71/-0)
- `Tensile/Tests/pre_checkin/test_hgemm_hpa_asm_nt.yaml`  (+69/-0)
- `Tensile/Tests/pre_checkin/test_hgemm_hpa_asm_tt.yaml`  (+66/-0)
- `Tensile/Source/Client.h`  (+31/-12)
- `Tensile/Tests/nightly/local_split_u/test_local_split_u.py`  (+14/-0)
- `Tensile/Tests/pre_checkin/test_pre_checkin.py`  (+12/-0)
- `Tensile/Common.py`  (+4/-4)
- `Tensile/ClientWriter.py`  (+3/-3)

## Key added lines (kernel files)

**`Tensile/ClientWriter.py`**
```
h += "const unsigned printTensorA=%x;\n" % int(globalParameters["PrintTensorA"])
h += "const unsigned printTensorB=%x;\n" % int(globalParameters["PrintTensorB"])
h += "const unsigned printTensorC=%x;\n" % int(globalParameters["PrintTensorC"])
```

**`Tensile/Common.py`**
```
globalParameters["PrintTensorA"] = 0          # Print TensorA after initialization
globalParameters["PrintTensorB"] = 0          # Print TensorB after initialization
globalParameters["PrintTensorC"] = 0          # Print TensorC.  0x1=after init; 0x2=after copy-back; 0x3=both
{"VectorAtomicWidth":         [ 1 ] },
```

**`Tensile/KernelWriterAssembly.py`**
```
self.do["ApplyAlpha"]  = True
def getLocalSplitUElementStep(self, kernel, isLds):
if isLds and \
kernel["VectorWidth"]*self.bpeCinternal >= 8 and \
```

**`Tensile/Source/Client.h`**
```
if (printTensorC & 0x1) {
printTensor("C_in", initialC, numIndicesC[problemTypeIdx],
if (printTensorC & 0x2) {
printTensor("C_result", deviceOnHostC, numIndicesC[problemTypeIdx],
```

**`Tensile/Source/MathTemplates.cpp`**
```
template<> TensileHalf tensileGetRandom<TensileHalf>() { return static_cast<TensileHalf>((rand()%7) - 3); }
```
