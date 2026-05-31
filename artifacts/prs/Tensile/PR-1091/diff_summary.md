# Diff summary

- **files changed:** 66
- **lines:** +1606 / -852
- **kernel-ish files:** 38

## Files (by churn)

- `Tensile/KernelWriterAssembly.py`  (+287/-261)
- `Tensile/KernelWriterBetaOnly.py`  (+246/-0)
- `Tensile/KernelWriterSource.py`  (+0/-216)
- `Tensile/SolutionWriter.py`  (+165/-50)
- `Tensile/KernelWriterConversion.py`  (+183/-0)
- `Tensile/SolutionStructs.py`  (+89/-39)
- `Tensile/Source/lib/source/ContractionSolution.cpp`  (+105/-7)
- `Tensile/KernelWriterBase.py`  (+100/-0)
- `Tensile/TensileCreateLibrary.py`  (+43/-46)
- `Tensile/KernelWriter.py`  (+0/-67)
- `Tensile/Source/client/include/DataInitializationTyped.hpp`  (+44/-19)
- `Tensile/Source/client/main.cpp`  (+57/-3)
- `Tensile/Contractions.py`  (+24/-17)
- `Tensile/BenchmarkProblems.py`  (+20/-20)
- `Tensile/Source/client/source/DataInitialization.cpp`  (+23/-12)

## Key added lines (kernel files)

**`HostLibraryTests/ProjectedPerformance_test.cpp`**
```
sizeMapping.globalAccumulation    = false;
sizeMapping.workspaceSizePerElemC = 0;
```

**`HostLibraryTests/TestData_test.cpp`**
```
auto datFiles  = data.glob(std::string("*.dat"));
```

**`HostLibraryTests/hip/RunGEMMKernel_test.cpp`**
```
catch(std::logic_error& exc)
```

**`HostLibraryTests/llvm/LLVMYAMLContraction_test.cpp`**
```
"  globalAccumulation: false\n"
"  workspaceSizePerElemC: 0\n"
"      globalAccumulation: false\n"
"      workspaceSizePerElemC: 0\n"
```

**`HostLibraryTests/testlib/include/TestData.hpp`**
```
boost::filesystem::path file(std::string const& filename) const;
boost::filesystem::path file(std::string const& filename, std::string const& extension) const;
```
