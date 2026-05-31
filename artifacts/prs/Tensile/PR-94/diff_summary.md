# Diff summary

- **files changed:** 29
- **lines:** +1329 / -773
- **kernel-ish files:** 12

## Files (by churn)

- `Tensile/Configs/rocblas_sgemm.yaml`  (+301/-114)
- `Tensile/KernelWriterSource.py`  (+83/-35)
- `Tensile/Configs/test_hgemm_scalar_tile_sizes.yaml`  (+90/-26)
- `Tensile/Configs/test_sgemm_scalar_tile_sizes.yaml`  (+90/-26)
- `Tensile/Configs/test_sgemm_vector_tile_sizes.yaml`  (+90/-26)
- `Tensile/SolutionStructs.py`  (+11/-98)
- `Tensile/Configs/rocblas_cgemm.yaml`  (+62/-36)
- `Tensile/Configs/rocblas_dgemm.yaml`  (+61/-35)
- `Tensile/Configs/rocblas_hgemm.yaml`  (+61/-35)
- `Tensile/Configs/rocblas_zgemm.yaml`  (+61/-35)
- `Tensile/Common.py`  (+63/-15)
- `Tensile/BenchmarkStructs.py`  (+21/-51)
- `Tensile/Configs/test_sgemm_scalar_branches.yaml`  (+37/-22)
- `Tensile/Configs/test_sgemm_vector_branches.yaml`  (+37/-22)
- `Tensile/Configs/sgemm_5760.yaml`  (+32/-24)

## Key added lines (kernel files)

**`Tensile/BenchmarkProblems.py`**
```
from Common import globalParameters, HR, pushWorkingPath, popWorkingPath, print1, print2, printExit, printWarning, ensur
print1("# Enumerating Solutions")
if globalParameters["PrintLevel"] >= 1:
progressBar = ProgressBar(maxPossibleSolutions)
```

**`Tensile/BenchmarkStructs.py`**
```
if name == joinName:
values = param[name]
localPermutations = len(values)
print2("JoinParameter %s has %u possibilities" % (joinName, localPermutations))
```

**`Tensile/ClientWriter.py`**
```
q = "" if os.name == "nt" else "\""
runScriptFile.write("%s & echo %s%s%s & echo %s# Configuring CMake for Client%s & echo %s%s%s\n" \
% (echoLine, q, HR, q, q, q, q, HR, q))
runScriptFile.write("%s & echo %s%s%s & echo %s# Building Client%s & echo %s%s%s\n" \
```

**`Tensile/Common.py`**
```
validWorkGroups = []
for numThreads in range(64, 1025, 64):
for nsg in [ 1, 2, 4, 8, 16, 32, 64 ]:
for sg0 in range(1, numThreads/nsg):
```

**`Tensile/KernelWriter.py`**
```
if kernel["EdgeType"] == "ShiftPtr":
if kernel["EdgeType"] == "ShiftPtr" and kernel["VectorWidth"] > 1:
```
