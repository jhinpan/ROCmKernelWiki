# Diff summary

- **files changed:** 37
- **lines:** +1022 / -281
- **kernel-ish files:** 18

## Files (by churn)

- `Tensile/KernelWriterSource.py`  (+392/-79)
- `Tensile/SolutionWriter.py`  (+154/-3)
- `Tensile/KernelWriter.py`  (+96/-41)
- `Tensile/SolutionStructs.py`  (+52/-17)
- `Tensile/Source/TensileConfigVersion.cmake`  (+66/-0)
- `Tensile/Common.py`  (+41/-5)
- `Tensile/Source/Client.h`  (+32/-12)
- `Tensile/KernelWriterAssembly.py`  (+21/-21)
- `Tensile/TensileCreateLibrary.py`  (+36/-2)
- `Tensile/Configs/rocblas_sgemm.yaml`  (+10/-10)
- `Tensile/YAMLIO.py`  (+12/-8)
- `Tensile/Configs/rocblas_cgemm.yaml`  (+6/-6)
- `Tensile/Configs/rocblas_dgemm.yaml`  (+6/-6)
- `Tensile/Configs/rocblas_hgemm.yaml`  (+6/-6)
- `Tensile/Configs/rocblas_zgemm.yaml`  (+6/-6)

## Key added lines (kernel files)

**`Tensile/BenchmarkProblems.py`**
```
import time
from Common import globalParameters, HR, pushWorkingPath, popWorkingPath, print1, print2, printExit, printWarning, ensur
currentTime = time.time()
elapsedTime = currentTime - startTime
```

**`Tensile/BenchmarkStructs.py`**
```
if hasParam("LocalSplitU", paramList):
splitUValues = getParamValues("LocalSplitU", paramList)
```

**`Tensile/ClientWriter.py`**
```
"TensileConfig.cmake",
"TensileConfigVersion.cmake"
h += "unsigned int dataInitTypeC = %s;\n" % globalParameters["DataInitTypeC"]
h += "unsigned int dataInitTypeAB = %s;\n" % globalParameters["DataInitTypeAB"]
```

**`Tensile/Common.py`**
```
from __init__ import __version__
import time
startTime = time.time()
globalParameters["DataInitTypeAB"] = 0 # 0=rand, 1=1, 2=serial, 3=0
```

**`Tensile/KernelWriter.py`**
```
if kernel["LocalSplitU"] > 1:
kStr += self.comment3("LocalSplitU Reduction")
kStr += self.comment("LocalSplitU: local write")
kStr += self.localSplitULocalWrite(kernel)
```
