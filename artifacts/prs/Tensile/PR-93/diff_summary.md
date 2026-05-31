# Diff summary

- **files changed:** 26
- **lines:** +863 / -484
- **kernel-ish files:** 14

## Files (by churn)

- `Tensile/LibraryLogic.py`  (+218/-80)
- `Tensile/SolutionStructs.py`  (+146/-24)
- `Tensile/KernelWriterSource.py`  (+45/-37)
- `Tensile/Source/Client.h`  (+39/-34)
- `Tensile/TensileCreateLibrary.py`  (+56/-14)
- `Tensile/Configs/sgemm.yaml`  (+0/-57)
- `Tensile/ClientWriter.py`  (+31/-25)
- `Tensile/Configs/test_sgemm_vector_branches.yaml`  (+26/-18)
- `Tensile/Configs/sgemm_5760.yaml`  (+24/-16)
- `Tensile/Configs/test_hgemm_scalar_branches.yaml`  (+24/-16)
- `Tensile/Configs/test_hgemm_scalar_tile_sizes.yaml`  (+24/-16)
- `Tensile/Configs/test_sgemm_scalar_branches.yaml`  (+24/-16)
- `Tensile/Configs/test_sgemm_scalar_tile_sizes.yaml`  (+24/-16)
- `Tensile/Configs/test_sgemm_vector_tile_sizes.yaml`  (+24/-16)
- `Tensile/TensileBenchmarkLibraryClient.py`  (+28/-10)

## Key added lines (kernel files)

**`Tensile/BenchmarkProblems.py`**
```
resultsIdxStr = "_%02u"%resultsFileIdx if len(resultsFileBaseList)>1 \
newResultsFileName = os.path.join(dataPath, "%s_%02u%s.csv" \
% (str(problemTypeObj), problemSizeGroupIdx, resultsIdxStr) )
newSolutionsFileName = os.path.join(dataPath, "%s_%02u%s.yaml" \
```

**`Tensile/BenchmarkStructs.py`**
```
configBenchmarkCommonParameters[0]["ProblemSizes"]
if len(paramValues) < 2 and paramName != "ProblemSizes":
print2("BenchmarkFinalParameters:")
```

**`Tensile/ClientWriter.py`**
```
indexOrder, exactLogic, rangeLogic) \
if forBenchmark:
h += "const unsigned int numProblems = %u;\n" \
% problemSizes.totalProblemSizes
```

**`Tensile/Common.py`**
```
"GroupShape":                 [ -64, -32, -16, -8, -4, -2,1,2,4,8,16,32,64],
"ThreadTileShape":            [ -64, -32, -16, -8, -4, -2,1,2,4,8,16,32,64],
"MacroTileShapeMin":          [ 1, 2, 4, 8, 16, 32, 64 ],
"MacroTileShapeMax":          [ 1, 2, 4, 8, 16, 32, 64 ],
```

**`Tensile/KernelWriterSource.py`**
```
kStr += "#define MAC(A,B,DST) mad(A,B,DST)"
if kernel["ProblemType"]["DataType"].isHalf():
kStr += "#define MAC(A,B,DST) DST = __hfma(A,B,DST)"
kStr += "#define MAC(A,B,DST) DST += A*B"
```
