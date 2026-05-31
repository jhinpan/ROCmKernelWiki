# Diff summary

- **files changed:** 29
- **lines:** +561 / -379
- **kernel-ish files:** 15

## Files (by churn)

- `Tensile/SolutionWriter.py`  (+86/-137)
- `Tensile/KernelWriterConversion.py`  (+103/-47)
- `Tensile/Source/lib/source/ContractionSolution.cpp`  (+91/-44)
- `Tensile/Tests/pre_checkin/mfma/hpa_hgemm_asm.yaml`  (+112/-0)
- `Tensile/KernelWriterAssembly.py`  (+61/-50)
- `Tensile/KernelWriterBetaOnly.py`  (+36/-54)
- `Tensile/SolutionStructs.py`  (+28/-20)
- `HostLibraryTests/hip/HipSolutionAdapter_test.cpp`  (+12/-11)
- `Tensile/Contractions.py`  (+9/-3)
- `HostLibraryTests/sample_library.yaml`  (+4/-4)
- `Tensile/Common.py`  (+6/-0)
- `Tensile/Source/Client.h`  (+3/-3)
- `HostLibraryTests/llvm/LLVMYAMLContraction_test.cpp`  (+2/-2)
- `Tensile/ClientWriter.py`  (+4/-0)
- `Tensile/Source/lib/include/Tensile/ContractionSolution.hpp`  (+2/-2)

## Key added lines (kernel files)

**`HostLibraryTests/ProjectedPerformance_test.cpp`**
```
sizeMapping.globalAccumulation    = 0;
```

**`HostLibraryTests/hip/HipSolutionAdapter_test.cpp`**
```
k.workGroupSize.x = 256;
k.workGroupSize.y = 1;
k.numWorkGroups.x = CeilDivide(desc.totalLogicalElements(), k.workGroupSize.x);
k.numWorkGroups.y = 1;
```

**`HostLibraryTests/llvm/LLVMYAMLContraction_test.cpp`**
```
"  globalAccumulation: 0\n"
"      globalAccumulation: 0\n"
```

**`Tensile/ClientWriter.py`**
```
maximumW = problemSizes.maxD * 32;
maximumW = max(maximumW, maxMN)
h += "size_t maxSizeW = %u;\n" % (maximumW)
h += "size_t maxSizeW;\n"
```

**`Tensile/Common.py`**
```
"GlobalSplitUAlgorithm":      ["SingleBuffer", "MultipleBuffer"],
{"GlobalSplitUAlgorithm":     [ "MultipleBuffer" ] },
```
