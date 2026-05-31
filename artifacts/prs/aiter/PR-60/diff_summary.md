# Diff summary

- **files changed:** 19
- **lines:** +889 / -67
- **kernel-ish files:** 15

## Files (by churn)

- `csrc/py_itfs_ck/rmsnorm_ck_kernels.cu`  (+276/-0)
- `op_tests/test_rmsnorm2dFusedAddQuant.py`  (+271/-0)
- `op_tests/test_rmsnorm2d.py`  (+111/-0)
- `ater/ops/rmsnorm.py`  (+77/-3)
- `ater/tuned_gemm.py`  (+33/-13)
- `csrc/include/rmsnorm.h`  (+42/-1)
- `op_tests/test_gemm.py`  (+31/-11)
- `gradlib/gradlib/GemmTuner.py`  (+10/-11)
- `ater/configs/untuned_gemm.csv`  (+7/-9)
- `csrc/pybind/rmsnorm_pybind.cu`  (+8/-2)
- `gradlib/gradlib/gemm_tuner.py`  (+6/-3)
- `ater/jit/optCompilerConfig.json`  (+5/-2)
- `csrc/kernels/cache_kernels.cu`  (+3/-3)
- `csrc/ck_gemm_a8w8/README.md`  (+2/-2)
- `csrc/py_itfs_ck/norm_kernels.cu`  (+2/-2)

## Key added lines (kernel files)

**`ater/ops/rmsnorm.py`**
```
def rms_norm_cu(
out: Tensor,
input: Tensor,
weight: Tensor,
```

**`ater/tuned_gemm.py`**
```
pd.set_option('display.max_colwidth', 100)
self.bestsols['kernelName'].fillna("")), "error: gradlib tune gemm not match the current environment, need re-tune!!!\n"
f'differece:\n{pd.concat([self.bestsols[['solidx','kernelName']], hipblasltKernelNames], axis=1)[hipblasltKernelNames !=
key = (ds['M'], ds['N'], ds['K'], ds['bias'], ds['dtype'], ds['outdtype'])
```

**`csrc/include/rmsnorm.h`**
```
torch::Tensor &weight, double epsilon);
torch::Tensor rmsnorm2d(torch::Tensor &input, torch::Tensor &weight,
double epsilon);
void rmsnorm2d_with_add(torch::Tensor &out,          // [m ,n]
```

**`csrc/kernels/cache_kernels.cu`**
```
const int64_t slot_idx = slot_mapping[token_idx];
if (token_idx >= num_tokens || slot_idx < 0)
```

**`csrc/py_itfs_ck/norm_kernels.cu`**
```
torch::Tensor &weight, // [1, n]
torch::Tensor &weight, // [1, n]
```
