# Diff summary

- **files changed:** 13
- **lines:** +112 / -67
- **kernel-ish files:** 13

## Files (by churn)

- `gradlib/gradlib/GemmTuner.py`  (+76/-49)
- `ater/tuned_gemm.py`  (+10/-5)
- `gradlib/csrc/hipbsolgemm.cu`  (+11/-0)
- `ater/test_common.py`  (+3/-3)
- `op_tests/test_smoothquant.py`  (+4/-2)
- `op_tests/test_gemm.py`  (+1/-1)
- `op_tests/test_gemm_a8w8.py`  (+1/-1)
- `op_tests/test_layernorm2d.py`  (+1/-1)
- `op_tests/test_layernorm2dFusedAddQuant.py`  (+1/-1)
- `op_tests/test_moe.py`  (+1/-1)
- `op_tests/test_moeTopkSoftmax.py`  (+1/-1)
- `op_tests/test_pa.py`  (+1/-1)
- `op_tests/test_transpose_add.py`  (+1/-1)

## Key added lines (kernel files)

**`ater/test_common.py`**
```
logger.info(f'avg: {avg} ms/iter from cuda.Event')
avg_name = '[avg ms/iter]'
```

**`ater/tuned_gemm.py`**
```
from ater import hipb_create_extension, hipb_mm, getHipblasltKernelName
if self.extensions_created is False:
rocb_create_extension()
hipb_create_extension()
```

**`gradlib/csrc/hipbsolgemm.cu`**
```
std::string getHipblasltKernelName(int solution_index)
std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(1);
std::vector<int> algoIndex(1);
algoIndex[0] = solution_index;
```

**`gradlib/gradlib/GemmTuner.py`**
```
from ater.test_common import perftest
self.topn = 20  # number of top solutions from each source
self.weights.t(),
bias=self.bias,
```

**`op_tests/test_gemm.py`**
```
from ater.test_common import checkAllclose, perftest
```
