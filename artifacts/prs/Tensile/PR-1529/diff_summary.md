# Diff summary

- **files changed:** 44
- **lines:** +3599 / -66
- **kernel-ish files:** 7

## Files (by churn)

- `Tensile/Tests/extended/direct_to_lds/dtl_dgemm.yaml`  (+2885/-0)
- `Tensile/Tests/pre_checkin/direct_to_lds/dtl_dgemm_lite.yaml`  (+330/-0)
- `tuning_docs/tensile_tuning.tex`  (+150/-5)
- `Tensile/Configs/rocblas_sgemm_example.yaml`  (+72/-0)
- `Tensile/Tests/pre_checkin/direct_to_lds/hgemm_asm_nn.yaml`  (+0/-48)
- `Tensile/KernelWriterAssembly.py`  (+32/-3)
- `Tensile/SolutionStructs.py`  (+26/-0)
- `Tensile/Source/lib/include/Tensile/MasterSolutionLibrary.hpp`  (+23/-1)
- `Tensile/TensileCreateLibrary.py`  (+16/-3)
- `Tensile/Common.py`  (+10/-4)
- `Tensile/Source/lib/source/Debug.cpp`  (+6/-1)
- `Tensile/Source/lib/include/Tensile/Debug.hpp`  (+3/-1)
- `Tensile/Tests/emulation/mfma/hpa_hgemm_asm.yaml`  (+4/-0)
- `Tensile/Tests/pre_checkin/mfma/hpa_hgemm_asm.yaml`  (+4/-0)
- `Tensile/Tests/pre_checkin/mfma/hpa_hgemm_general_batch_asm.yaml`  (+4/-0)

## Key added lines (kernel files)

**`Tensile/Common.py`**
```
globalParameters["HipClangVersion"] = "0.0.0"
"SplitGlobalRead":            [1, 2, 4, 8],
{"StoreVectorWidth":          [ -1 ] },
{"WaveSeparateGlobalReadA":   [ 0 ] },
```

**`Tensile/KernelWriterAssembly.py`**
```
kStr += vectorStaticRemainder(dummy, dividendReg, "Serial", kernel["WavefrontSize"], tmpVgpr, tmpSgpr)
splitRead = kernel["SplitGlobalRead"]
if divisor > kernel["WavefrontSize"]:
splitRead = 1
```

**`Tensile/SolutionStructs.py`**
```
@staticmethod
def getDivisorName(state, tC):
if state["GlobalReadCoalesceGroup{}".format(tC)]:
if state["GlobalReadCoalesceVector{}".format(tC)]:
```

**`Tensile/Source/lib/include/Tensile/Debug.hpp`**
```
bool printSolutionSelectionTime() const;
```

**`Tensile/Source/lib/include/Tensile/MasterSolutionLibrary.hpp`**
```
if(Debug::Instance().printSolutionSelectionTime())
auto start  = std::chrono::steady_clock::now();
auto result = findBestSolution_runner(problem, hardware, fitness);
auto end    = std::chrono::steady_clock::now();
```
