# Diff summary

- **files changed:** 40
- **lines:** +296 / -316
- **kernel-ish files:** 17

## Files (by churn)

- `Tensile/LibraryLogic.py`  (+59/-59)
- `Tensile/TensileCreateLibrary.py`  (+36/-40)
- `Tensile/BenchmarkProblems.py`  (+32/-32)
- `Tensile/Tensile.py`  (+29/-29)
- `Tensile/BenchmarkStructs.py`  (+22/-22)
- `Tensile/Common.py`  (+22/-18)
- `Tensile/TensileRetuneLibrary.py`  (+15/-15)
- `Tensile/KernelWriterAssembly.py`  (+14/-14)
- `Tensile/TensileLibLogicToYaml.py`  (+14/-14)
- `Tensile/SolutionStructs.py`  (+10/-10)
- `Tensile/TensileUpdateLibrary.py`  (+10/-10)
- `Tensile/KernelWriter.py`  (+8/-10)
- `Tensile/TensileClientConfig.py`  (+9/-9)
- `Tensile/ClientExecutable.py`  (+5/-5)
- `Tensile/TensileBenchmarkCluster.py`  (+3/-3)

## Key added lines (kernel files)

**`Tensile/AsmRegisterPool.py`**
```
from .Common import tPrint, printExit, printWarning
tPrint(3, "total vgpr count: %u\n"%self.size())
```

**`Tensile/BenchmarkProblems.py`**
```
from .Common import globalParameters, HR, pushWorkingPath, popWorkingPath, tPrint, \
tPrint(1, "# Enumerating Solutions")
tPrint(1, "rejecting solution " + str(solutionObject))
tPrint(1, "# Processing custom kernel {}".format(kernelName))
```

**`Tensile/BenchmarkStructs.py`**
```
from .Common import tPrint, printExit, \
tPrint(3, "# BenchmarkProcess beginning {}".format(self.problemType))
tPrint(3, "")
tPrint(3, "####################################################################")
```

**`Tensile/ClientExecutable.py`**
```
from .Common import globalParameters, tPrint, supportedCompiler
Common.tPrint(3, ' '.join(args))
tPrint(3, out)
Common.tPrint(3, ' '.join(args))
```

**`Tensile/ClientWriter.py`**
```
from .Common import globalParameters, pushWorkingPath, popWorkingPath, tPrint, printExit, printWarning, ClientExecutionL
tPrint(1, "LogicFiles: %s" % logicFiles)
```
