# Diff summary

- **files changed:** 62
- **lines:** +1768 / -794
- **kernel-ish files:** 40

## Files (by churn)

- `Tensile/KernelWriterAssembly.py`  (+497/-181)
- `Tensile/SolutionWriter.py`  (+98/-141)
- `Tensile/SolutionStructs.py`  (+102/-105)
- `Tensile/Source/lib/source/ContractionSolution.cpp`  (+130/-49)
- `Tensile/KernelWriter.py`  (+131/-39)
- `Tensile/KernelWriterConversion.py`  (+103/-47)
- `Tensile/Source/client/source/Reference.cpp`  (+77/-65)
- `Tensile/KernelWriterSource.py`  (+95/-38)
- `Tensile/Tests/pre_checkin/mfma/hpa_hgemm_asm.yaml`  (+112/-0)
- `Tensile/KernelWriterBetaOnly.py`  (+36/-54)
- `Tensile/Source/lib/source/ContractionProblem.cpp`  (+47/-0)
- `Tensile/Source/lib/include/Tensile/ContractionProblemPredicates.hpp`  (+28/-1)
- `Tensile/TensileCreateLibrary.py`  (+23/-5)
- `Tensile/Tests/pre_checkin/bfloat16/bfloat16_hpa_source_nn.yaml`  (+25/-2)
- `Tensile/Tests/pre_checkin/bfloat16/bfloat16_hpa_source_nt.yaml`  (+25/-2)

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

**`Tensile/ClientWriter.py`**
```
maximumW = problemSizes.maxD * 32;
maximumW = max(maximumW, maxMN)
h += "size_t maxSizeW = %u;\n" % (maximumW)
h += "size_t maxSizeW;\n"
```
