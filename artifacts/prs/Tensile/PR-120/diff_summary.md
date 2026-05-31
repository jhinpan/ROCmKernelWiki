# Diff summary

- **files changed:** 32
- **lines:** +1076 / -1463
- **kernel-ish files:** 15

## Files (by churn)

- `Tensile/KernelWriterAssembly.py`  (+466/-547)
- `Tensile/Configs/sgemm_asm.yaml`  (+252/-0)
- `Tensile/Configs/rocblas_sgemm.yaml`  (+96/-95)
- `Tensile/Configs/sgemm_gfx803.yaml`  (+0/-177)
- `Tensile/Configs/sgemm_gfx900.yaml`  (+0/-177)
- `Tensile/TensileCreateLibrary.py`  (+59/-80)
- `Tensile/LibraryLogic.py`  (+30/-60)
- `Tensile/SolutionStructs.py`  (+12/-49)
- `Tensile/Common.py`  (+35/-15)
- `Tensile/SolutionWriter.py`  (+35/-13)
- `Jenkinsfile`  (+10/-26)
- `Tensile/BenchmarkProblems.py`  (+18/-13)
- `Tensile/KernelWriter.py`  (+18/-12)
- `Tensile/Configs/test_hgemm.yaml`  (+1/-23)
- `Tensile/Configs/test_hgemm_vectors.yaml`  (+1/-23)

## Key added lines (kernel files)

**`Tensile/BenchmarkProblems.py`**
```
from Common import globalParameters, HR, pushWorkingPath, popWorkingPath, print1, print2, printExit, printWarning, ensur
kernelsBetaOnly = []
solutionKernelsBetaOnly = solution.getKernelsBetaOnly()
for kernel in solutionKernelsBetaOnly:
```

**`Tensile/ClientWriter.py`**
```
runScriptFile.write("%s && echo %s%s%s && echo %s# Configuring CMake for Client%s && echo %s%s%s\n" \
runScriptFile.write("%s && echo %s%s%s && echo %s# Building Client%s && echo %s%s%s\n" \
runScriptFile.write("%s && echo %s%s%s && echo %s# Library Client:%s && echo %s# %s%s && %s\n" \
% (echoLine, q, HR, q, q, q, q, executablePath, q, executablePath) )
```

**`Tensile/Common.py`**
```
from subprocess import Popen, PIPE
globalParameters["SupportedISA"] = [(8,0,3), (9,0,0)]
globalParameters["CurrentISA"] = (0,0,0)
validISA = [(0,0,0)]
```

**`Tensile/KernelWriter.py`**
```
from Common import globalParameters, pushWorkingPath, popWorkingPath, printWarning, printExit, print1, print2, HR
beforeFunctionSignature = kStr
kStr = ""
afterFunctionSignature = kStr
```

**`Tensile/KernelWriterAssembly.py`**
```
from Common import globalParameters, print1, print2, printExit, printWarning
class RegisterPool:
statusUnAvailable = 0
statusAvailable = 1
```
