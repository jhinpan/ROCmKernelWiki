# Diff summary

- **files changed:** 23
- **lines:** +949 / -130
- **kernel-ish files:** 6

## Files (by churn)

- `Tensile/KernelWriter.py`  (+91/-40)
- `Tensile/Configs/test_sgemm_scalar_branches.yaml`  (+127/-0)
- `Tensile/Configs/test_sgemm_scalar_load_patterns.yaml`  (+127/-0)
- `Tensile/Configs/test_sgemm_scalar_tile_sizes.yaml`  (+127/-0)
- `Tensile/Configs/test_sgemm_vector_branches.yaml`  (+127/-0)
- `Tensile/Configs/test_sgemm_vector_load_patterns.yaml`  (+127/-0)
- `Tensile/Configs/test_sgemm_vector_tile_sizes.yaml`  (+127/-0)
- `Tensile/LibraryLogic.py`  (+15/-32)
- `Tensile/Configs/sgemm_5760.yaml`  (+25/-21)
- `Tensile/Configs/rocblas_zgemm.yaml`  (+9/-9)
- `Tensile/Configs/rocblas_sgemm.yaml`  (+9/-5)
- `Tensile/ClientWriter.py`  (+10/-3)
- `Jenkinsfile`  (+8/-2)
- `Tensile/Configs/rocblas_cgemm.yaml`  (+5/-5)
- `Tensile/Configs/rocblas_dgemm.yaml`  (+5/-5)

## Key added lines (kernel files)

**`Tensile/ClientWriter.py`**
```
runScriptFile.write(" -DTensile_ROOT=%s" \
% os.path.join(globalParameters["ScriptPath"], "..") )
runScriptFile.write(" -DTensile_KERNEL_LANGUAGE=%s" \
% globalParameters["KernelLanguage"])
```

**`Tensile/Common.py`**
```
globalParameters["CMakeCXXFlags"] = ""
globalParameters["CMakeCFlags"] = ""
globalParameters["NumElementsToValidate"] = 128
globalParameters["ValidationMaxToPrint"] = 4
```

**`Tensile/KernelWriter.py`**
```
if readTileDimVectorA or readUnrollDimVectorA:
kStr += "%sglobalReadA_%u_%u%s = (%sVECTOR_TYPE *)( ((%sDATA_TYPE *)globalReadA_%u_%u%s) + ((%s) strideA%s)*DEPTHU);%s" 
% (indent, \
para, perp, \
```

**`Tensile/LibraryLogic.py`**
```
print1("Solutions Used:")
for i in range(0, len(logicAnalyzer.solutions)):
print1("(%2u) %s" % (i, Solution.getNameFull(logicAnalyzer.solutions[i])))
print self.data
```
