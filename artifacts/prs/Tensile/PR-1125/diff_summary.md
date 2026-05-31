# Diff summary

- **files changed:** 31
- **lines:** +614 / -155
- **kernel-ish files:** 17

## Files (by churn)

- `Tensile/KernelWriterAssembly.py`  (+259/-79)
- `Tensile/KernelWriter.py`  (+111/-28)
- `Tensile/KernelWriterSource.py`  (+48/-26)
- `Tensile/Source/lib/source/ContractionProblem.cpp`  (+54/-0)
- `Tensile/SolutionStructs.py`  (+24/-13)
- `Tensile/Source/lib/source/ContractionSolution.cpp`  (+29/-5)
- `Tensile/Source/lib/include/Tensile/ContractionProblemPredicates.hpp`  (+27/-0)
- `Tensile/Components/Signature.py`  (+14/-4)
- `Tensile/Source/lib/include/Tensile/ContractionProblem.hpp`  (+9/-0)
- `HostLibraryTests/sample_library.yaml`  (+8/-0)
- `Tensile/Common.py`  (+7/-0)
- `Tensile/Source/lib/include/Tensile/Serialization/ContractionPredicates.hpp`  (+7/-0)
- `Tensile/Contractions.py`  (+5/-0)
- `HostLibraryTests/ProjectedPerformance_test.cpp`  (+3/-0)
- `Tensile/Tests/pre_checkin/regression/persistent_kernel.yaml`  (+3/-0)

## Key added lines (kernel files)

**`HostLibraryTests/ProjectedPerformance_test.cpp`**
```
sizeMapping.persistentKernel           = 0;
sizeMapping.persistentKernelAlongBatch = false;
```

**`HostLibraryTests/llvm/LLVMYAMLContraction_test.cpp`**
```
"  persistentKernelAlongBatch: false\n"
"      persistentKernelAlongBatch: false\n"
```

**`Tensile/Common.py`**
```
"PersistentKernelAlongBatch": [False,True],
{"PersistentKernelAlongBatch":[ False ] },    # May be default True is better ?
```

**`Tensile/Components/Signature.py`**
```
if kernel["PersistentKernel"]:
kStr += self.v2Argument("MagicNumberProblemNumGroupTiles0",     '4',    '4',      "ByValue",      "U32"); ka_size += 4
kStr += self.v2Argument(              "GridNumWorkGroups0",     '4',    '4',      "ByValue",      "U32"); ka_size += 4
if kernel["PersistentKernelAlongBatch"]:
```

**`Tensile/Contractions.py`**
```
if 'PersistentKernel' in state and state['PersistentKernel']:
rv += [cls("PersistentKernelCheck", value = True)]
'persistentKernelAlongBatch',
persistentKernelAlongBatch   = d['PersistentKernelAlongBatch'] if 'PersistentKernelAlongBatch' in d else False,
```
