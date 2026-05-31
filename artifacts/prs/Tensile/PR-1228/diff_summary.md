# Diff summary

- **files changed:** 52
- **lines:** +2887 / -782
- **kernel-ish files:** 32

## Files (by churn)

- `Tensile/KernelWriterAssembly.py`  (+641/-575)
- `Tensile/Tests/pre_checkin/mfma/hpa_hgemm_general_batch_asm.yaml`  (+299/-0)
- `Tensile/Tests/pre_checkin/dgemm_general_batch_asm.yaml`  (+276/-0)
- `Tensile/Source/client/include/DataInitializationTyped.hpp`  (+169/-42)
- `Tensile/Source/lib/source/ContractionProblem.cpp`  (+183/-17)
- `Tensile/Tests/pre_checkin/mfma/hpa_bfloat16_general_batch_gemm_asm.yaml`  (+200/-0)
- `Tensile/Source/lib/include/Tensile/ContractionProblem.hpp`  (+192/-6)
- `Tensile/Tests/pre_checkin/mfma/sgemm_general_batch_asm.yaml`  (+185/-0)
- `Tensile/KernelWriterSource.py`  (+68/-30)
- `Tensile/Tests/pre_checkin/sgemm_general_batch_asm_nn.yaml`  (+94/-0)
- `Tensile/Source/lib/source/ContractionSolution.cpp`  (+66/-16)
- `Tensile/Tests/pre_checkin/hgemm_general_batch_asm_nn.yaml`  (+82/-0)
- `Tensile/Tests/pre_checkin/hgemm_general_batch_hpa_asm_nn.yaml`  (+82/-0)
- `Tensile/Source/lib/include/Tensile/ContractionProblemPredicates.hpp`  (+47/-27)
- `Tensile/KernelWriterBetaOnly.py`  (+58/-14)

## Key added lines (kernel files)

**`HostLibraryTests/client/DataInitialization_test.cpp`**
```
args.insert({"offset-a", val((size_t)0, false)});
args.insert({"offset-b", val((size_t)0, false)});
args.insert({"offset-c", val((size_t)0, false)});
args.insert({"offset-d", val((size_t)0, false)});
```

**`HostLibraryTests/hip/HipSolutionAdapter_test.cpp`**
```
k.args.append<unsigned int>("offsetD", desc.offset());
k.args.append<unsigned int>("offsetC", desc.offset());
k.args.append<unsigned int>("offsetD", desc.offset());
k.args.append<unsigned int>("offsetC", desc.offset());
```

**`Tensile/BenchmarkStructs.py`**
```
validParameters, defaultSolutionSummationSizes, globalParameters
def checkCDBufferAndStrides(problemType, problemSizes, isCEqualD):
if isCEqualD and problemType["OperationType"] == "GEMM":
for problem in problemSizes.problems:
```

**`Tensile/ClientWriter.py`**
```
param('high-precision-accumulate', problemType.highPrecisionAccumulate)
param('strided-batched', problemType.stridedBatched)
param("offset-a",                 globalParameters["BufferOffsetA"])
param("offset-b",                 globalParameters["BufferOffsetB"])
```

**`Tensile/Common.py`**
```
globalParameters["CEqualD"] = False               # Set to true if testing for the case where the pointer to C is the sa
globalParameters["BufferOffsetA"] = 0             # data offset of buffer A
globalParameters["BufferOffsetB"] = 0             # data offset of buffer B
globalParameters["BufferOffsetC"] = 0             # data offset of buffer C
```
