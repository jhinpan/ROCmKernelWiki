# Diff summary

- **files changed:** 17
- **lines:** +441 / -92
- **kernel-ish files:** 8

## Files (by churn)

- `Tensile/KernelWriterAssembly.py`  (+134/-71)
- `Tensile/Tests/nightly/global_split_u/hgemm_gsu.yaml`  (+80/-0)
- `Tensile/Tests/nightly/global_split_u/sgemm_gsu.yaml`  (+78/-0)
- `Tensile/Configs/rocblas_hgemm_asm_single_kernel.yaml`  (+64/-0)
- `Tensile/SolutionStructs.py`  (+30/-4)
- `Tensile/Configs/rocblas_sgemm_asm_full.yaml`  (+18/-0)
- `Tensile/Tests/nightly/local_split_u/test_local_split_u.py`  (+7/-7)
- `Tensile/Common.py`  (+5/-4)
- `Tensile/Configs/rocblas_dgemm_asm_lite.yaml`  (+8/-0)
- `Tensile/Tests/nightly/global_split_u/test_global_split_u.py`  (+8/-0)
- `Tensile/ClientWriter.py`  (+2/-2)
- `Tensile/Source/Client.h`  (+2/-2)
- `Tensile/Tests/create_tests.py`  (+3/-0)
- `Tensile/Tests/nightly/local_split_u/dgemm_lsu.yaml`  (+1/-1)
- `Tensile/Tests/nightly/local_split_u/sgemm_lsu.yaml`  (+1/-1)

## Key added lines (kernel files)

**`Tensile/ClientWriter.py`**
```
h += "    if (strideA != std::numeric_limits<unsigned int>::max())  strideA%u%s = strideA;\n" % (lastStrideA-1, indexCha
h += "    if (strideB != std::numeric_limits<unsigned int>::max())  strideB%u%s = strideB;\n" % (lastStrideB-1, indexCha
```

**`Tensile/Common.py`**
```
"VectorAtomicWidth":          [ -1, 1, 2 ] ,
{"VectorAtomicWidth":         [ -1 ] },
```

**`Tensile/KernelWriterAssembly.py`**
```
return 1  # flat vector atomic is not tested
regsPerElement = 2 if kernel["BufferStore"] else 3
numVgprsPerDataPerVI = (regsPerElement*self.bpeCexternal)/self.bpr
atomicW = min(gwvw, kernel["VectorAtomicWidth"])
```

**`Tensile/SolutionStructs.py`**
```
supported = \
state["ProblemType"]["DataType"].isSingle() or \
(state["KernelLanguage"] == "Assembly" and \
(state["ProblemType"]["DataType"].isHalf() or \
```

**`Tensile/Source/Client.h`**
```
v = std::fabs(double(v));
v = s*std::fabs(double(v));
```
