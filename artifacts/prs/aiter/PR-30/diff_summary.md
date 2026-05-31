# Diff summary

- **files changed:** 13
- **lines:** +202 / -112
- **kernel-ish files:** 11

## Files (by churn)

- `ater/ops/gemm_op_a8w8.py`  (+61/-12)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`  (+42/-26)
- `ater/configs/a8w8_tuned_gemm.csv`  (+27/-27)
- `op_tests/test_gemm_a8w8.py`  (+22/-20)
- `csrc/ck_gemm_a8w8/gemm_a8w8.cu`  (+11/-6)
- `csrc/ck_gemm_a8w8/gen_instances.py`  (+10/-7)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.cu`  (+9/-4)
- `csrc/ck_gemm_a8w8/include/gemm_a8w8_common.cuh`  (+6/-4)
- `csrc/ck_gemm_a8w8/include/gemm_a8w8.h`  (+4/-2)
- `csrc/rocm_ops.cpp`  (+4/-1)
- `csrc/ck_gemm_a8w8/README.md`  (+2/-1)
- `csrc/pybind/gemm_a8w8_pybind.cu`  (+2/-1)
- `csrc/pybind/gemm_a8w8_tune_pybind.cu`  (+2/-1)

## Key added lines (kernel files)

**`ater/ops/gemm_op_a8w8.py`**
```
md_name="module_gemm_a8w8",
splitK = 0
sub_m: Optional[int] = 128,
sub_n: Optional[int] = 128,
```

**`csrc/ck_gemm_a8w8/gemm_a8w8.cu`**
```
torch::Tensor &, torch::Tensor &,
torch::Tensor &, std::optional<torch::Tensor>,
std::optional<torch::Tensor> bias,
int splitK)
```

**`csrc/ck_gemm_a8w8/gemm_a8w8_tune.cu`**
```
torch::Tensor &, torch::Tensor &,
torch::Tensor &, std::optional<torch::Tensor>,
"Kernel id " + std::to_string(id)  +" is out of range!");
int kernelId,
```

**`csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`**
```
tunedf = pd.DataFrame(columns=["M", "N", "K", "kernelId", "splitK", "us", "kernelName"])
def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
ater.gemm_a8w8_tune(x, weight, x_scale, w_scale, out, kernel_id, splitK)
def tune_gemm(m, n, k, useSplitK = False):
```

**`csrc/ck_gemm_a8w8/gen_instances.py`**
```
std::optional<torch::Tensor> bias,
int KBatch)
bool pad = (M % {k.MPerBLOCK} != 0) || (N % {k.NPerBLOCK} != 0) || (K % ({k.KPerBLOCK} * KBatch) != 0);
return gemm_a8w8_mma_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
```
