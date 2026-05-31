# Diff summary

- **files changed:** 24
- **lines:** +1539 / -1538
- **kernel-ish files:** 9

## Files (by churn)

- `Tensile/Configs/test_hgemm.yaml`  (+333/-0)
- `Tensile/Configs/test_hgemm_vectors.yaml`  (+333/-0)
- `Tensile/Configs/test_sgemm.yaml`  (+333/-0)
- `Tensile/Configs/test_sgemm_vectors.yaml`  (+333/-0)
- `Tensile/Configs/test_hgemm_scalar_tile_sizes.yaml`  (+0/-190)
- `Tensile/Configs/test_sgemm_scalar_tile_sizes.yaml`  (+0/-190)
- `Tensile/Configs/test_sgemm_vector_tile_sizes.yaml`  (+0/-190)
- `Tensile/BenchmarkStructs.py`  (+85/-85)
- `Tensile/Configs/test_sgemm_scalar_branches.yaml`  (+0/-141)
- `Tensile/Configs/test_sgemm_vector_branches.yaml`  (+0/-141)
- `Tensile/Configs/test_hgemm_scalar_branches.yaml`  (+0/-137)
- `Tensile/Configs/test_hgemm_scalar_load_patterns.yaml`  (+0/-137)
- `Tensile/Configs/test_sgemm_vector_load_patterns.yaml`  (+0/-137)
- `Tensile/Configs/test_sgemm_scalar_load_patterns.yaml`  (+0/-133)
- `Tensile/Configs/tensor_contraction.yaml`  (+41/-1)

## Key added lines (kernel files)

**`Tensile/BenchmarkProblems.py`**
```
print1("# Already benchmarked; skipping.")
if globalParameters["PrintLevel"] >= 1:
print1("# Adding Results to Solution Database")
progressBar = ProgressBar(len(results))
```

**`Tensile/BenchmarkStructs.py`**
```
if len(self.joinParameters) > 1:
if hasParam(joinName, self.singleValueParameters):
elif hasParam(joinName, self.forkParameters):
for param in self.forkParameters:
```

**`Tensile/Common.py`**
```
globalParameters["SyncsPerBenchmark"] = 1
{"ThreadTile":                [ [4,4] ] },
{"DepthU":                    [ 16 ] },
defaultForkParameters = []
```

**`Tensile/KernelWriter.py`**
```
kStr += self.calculateLoopNumIter(kernel, i)
if not globalParameters["MergeFiles"]:
fileString += "\n"
fileString += "#include \"%s.h\"\n" % kernelName
```

**`Tensile/KernelWriterSource.py`**
```
kStr += "__device__ inline void atomicAddType(%s%sfloat *fPtr, float operand) {%s" \
if kernel["ProblemType"]["UseBeta"]:
kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG,BETA) atomicAddType(&(DST), (ALPHA)*(REG));"
kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG) atomicAddType(&(DST), (ALPHA)*(REG));"
```
