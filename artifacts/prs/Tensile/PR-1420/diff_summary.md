# Diff summary

- **files changed:** 90 (diff was byte-capped; summary is partial)
- **lines:** +130 / -5503
- **kernel-ish files:** 15

## Files (by churn)

- `Tensile/Source/Client.h`  (+0/-3205)
- `Tensile/Source/MathTemplates.cpp`  (+0/-814)
- `Tensile/Source/Client.cpp`  (+0/-394)
- `Tensile/Source/DeviceStats.h`  (+0/-329)
- `Tensile/Source/CMakeLists.txt`  (+46/-191)
- `Tensile/ClientWriter.py`  (+24/-174)
- `Tensile/KernelWriterAssembly.py`  (+2/-98)
- `Tensile/BenchmarkProblems.py`  (+24/-62)
- `Tensile/SolutionSelectionLibrary.py`  (+0/-65)
- `.dir-locals.el`  (+0/-30)
- `Tensile/Common.py`  (+11/-17)
- `Tensile/KernelWriter.py`  (+10/-9)
- `Tensile/SolutionStructs.py`  (+4/-11)
- `Tensile/Source/MathTemplates.h`  (+0/-15)
- `Tensile/LibraryLogic.py`  (+3/-6)

## Key added lines (kernel files)

**`Tensile/BenchmarkProblems.py`**
```
from .ClientWriter import runClient, writeClientConfig
pushWorkingPath("source")
filesToCopy = [
"TensileTypes.h",
```

**`Tensile/ClientWriter.py`**
```
from .Common import globalParameters, pushWorkingPath, popWorkingPath, print1, printExit, CHeader, printWarning, listToI
createLibraryScript = getBuildClientLibraryScript(stepBaseDir, libraryLogicPath)
clientExe = ClientExecutable.getClientExecutable(clientBuildDir)
args = [clientExe, iniFile]
```

**`Tensile/Common.py`**
```
globalParameters["DataInitTypeAB"] = 3
globalParameters["DataInitTypeA"] = -1
globalParameters["DataInitTypeB"] = -1
globalParameters["DataInitTypeC"]  = 3
```

**`Tensile/GenerateSummations.py`**
```
"--merge-files", "--new-client-only", "--no-short-file-names", "--no-library-print-debug", \
"--architecture=all", "--code-object-version=V3", "--cxx-compiler=hipcc", "--library-format=yaml", \
```

**`Tensile/KernelWriter.py`**
```
self.getSingleCodeObjectFile(kernel)
return (0, "")
if globalParameters["PrintSolutionRejectionReason"]:
print(exc)
```
