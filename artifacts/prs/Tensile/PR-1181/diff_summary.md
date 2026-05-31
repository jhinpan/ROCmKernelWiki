# Diff summary

- **files changed:** 56 (diff was byte-capped; summary is partial)
- **lines:** +1818 / -959
- **kernel-ish files:** 42

## Files (by churn)

- `Tensile/KernelWriterAssembly.py`  (+487/-170)
- `Tensile/SolutionWriter.py`  (+98/-141)
- `Tensile/SolutionStructs.py`  (+102/-105)
- `Tensile/Source/lib/source/ContractionSolution.cpp`  (+131/-49)
- `Tensile/KernelWriter.py`  (+131/-39)
- `Tensile/BenchmarkStructs.py`  (+111/-51)
- `Tensile/KernelWriterConversion.py`  (+103/-47)
- `Tensile/Source/client/source/Reference.cpp`  (+77/-65)
- `Tensile/ClientWriter.py`  (+83/-54)
- `Tensile/KernelWriterSource.py`  (+95/-38)
- `Tensile/BenchmarkProblems.py`  (+69/-49)
- `Tensile/KernelWriterBetaOnly.py`  (+36/-54)
- `Tensile/LibraryLogic.py`  (+25/-32)
- `Tensile/Source/lib/source/ContractionProblem.cpp`  (+47/-0)
- `Tensile/Source/lib/include/Tensile/ContractionProblemPredicates.hpp`  (+28/-1)

## Key added lines (kernel files)

**`HostLibraryTests/ProjectedPerformance_test.cpp`**
```
sizeMapping.persistentKernel           = 0;
sizeMapping.persistentKernelAlongBatch = false;
sizeMapping.globalAccumulation    = 0;
```

**`HostLibraryTests/hip/HipSolutionAdapter_test.cpp`**
```
k.workGroupSize.x = 256;
k.workGroupSize.y = 1;
k.numWorkGroups.x = CeilDivide(desc.totalLogicalElements(), k.workGroupSize.x);
k.numWorkGroups.y = 1;
```

**`HostLibraryTests/hip/RunGEMMKernel_test.cpp`**
```
TEST_P(RunGEMMKernelTest, TestAlphaZeroSigned)
auto param     = GetParam();
auto typedTest = std::get<0>(param);
typedTest->OverrideAlpha(std::copysign(0.0, -1.0));
```

**`HostLibraryTests/llvm/LLVMYAMLContraction_test.cpp`**
```
"  globalAccumulation: 0\n"
"  persistentKernelAlongBatch: false\n"
"      globalAccumulation: 0\n"
"      persistentKernelAlongBatch: false\n"
```

**`Tensile/BenchmarkProblems.py`**
```
from .BenchmarkStructs import BenchmarkProcess, constructForkPermutations
def generateForkedSolutions (problemType, hardcodedParameters, benchmarkPermutations, winners=None, initialSolutionParam
"""this creates a set or solutions based on the forked parameters using
a set of common parameters from which to fork from
```
