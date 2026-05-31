# Diff summary

- **files changed:** 24 (diff was byte-capped; summary is partial)
- **lines:** +3898 / -471
- **kernel-ish files:** 16

## Files (by churn)

- `Tensile/ReplacementKernels/Cijk_Alik_Bljk_BBH_MT64x128x32_SE_K1.s.txt`  (+1018/-0)
- `Tensile/KernelWriterAssembly.py`  (+700/-248)
- `Tensile/Configs/rocblas_cgemm_asm_lite.yaml`  (+606/-0)
- `Tensile/Configs/rocblas_cgemm_hip_lite.yaml`  (+364/-0)
- `Tensile/Source/Client.h`  (+297/-12)
- `Tensile/ClientWriter.py`  (+154/-40)
- `Tensile/KernelWriterSource.py`  (+134/-56)
- `Tensile/SolutionSelectionLibrary.py`  (+123/-0)
- `Tensile/Configs/rocblas_hpa_bfloat16_asm_single_kernel.yaml`  (+100/-0)
- `Tensile/SolutionStructs.py`  (+63/-33)
- `Tensile/BenchmarkProblems.py`  (+56/-11)
- `Tensile/Common.py`  (+63/-3)
- `Tensile/SolutionLibrary.py`  (+61/-3)
- `Tensile/LibraryLogic.py`  (+47/-11)
- `Tensile/KernelWriter.py`  (+32/-10)

## Key added lines (kernel files)

**`Tensile/BenchmarkProblems.py`**
```
enableTileSelection = benchmarkProcess.problemType["TileAwareSelection"]
shortName, filesToCopy, benchmarkProcess.solutionSummationSizes)
removeSolutions = []
for i in range(0, len(solutions)):
```

**`Tensile/BenchmarkStructs.py`**
```
validParameters, defaultSolutionSummationSizes
self.solutionSummationSizes = []
self.solutionSummationSizes = defaultSolutionSummationSizes
if "SolutionSummationSizes" in problemSizesDict:
```

**`Tensile/ClientWriter.py`**
```
solutionSummationSizes = None
functions, solutionSummationSizes, stepBaseDir)
enableTileSelection = False
runScriptName = writeRunScript(path, libraryLogicPath, forBenchmark, enableTileSelection)
```

**`Tensile/Common.py`**
```
globalParameters["SupportedISA"] = [(8,0,3), (9,0,0), (9,0,6), (9,0,8)]             # assembly kernels writer supports t
globalParameters["CxxCompiler"] = "hcc"
"MinVgprNumber":                list(range(0,256)),
"MaxVgprNumber":                list(range(0,257)),
```

**`Tensile/Contractions.py`**
```
'ideals']
if 'Ideals' in d:
rv.ideals = d['Ideals']
rv.ideals = {}
```
