# Diff summary

- **files changed:** 12
- **lines:** +64 / -41
- **kernel-ish files:** 12

## Files (by churn)

- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle.cu`  (+10/-7)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile.cu`  (+10/-7)
- `aiter/ops/gemm_op_a8w8.py`  (+8/-3)
- `csrc/ck_gemm_a8w8_bpreshuffle/gen_instances.py`  (+7/-4)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gen_instances.py`  (+7/-4)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.cu`  (+4/-3)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile_tune.cu`  (+4/-3)
- `csrc/cktile_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle_cktile_common.cuh`  (+3/-3)
- `csrc/include/rocm_ops.hpp`  (+4/-2)
- `csrc/ck_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle_common.cuh`  (+3/-2)
- `csrc/cktile_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle_cktile.h`  (+2/-2)
- `csrc/ck_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle.h`  (+2/-1)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a8w8.py`**
```
splitK: int = 0,
splitK: int = 0,
splitK: int = 0,
splitK: int = 0,
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle.cu`**
```
torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, int)>;
torch::Tensor& Y,
int splitK)
TORCH_CHECK(splitK >= 0, "splitK must be non-negative!");
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.cu`**
```
torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, int)>;
TORCH_CHECK(splitK >= 0, "splitK must be non-negative!");
int KBatch = 1 << splitK;
blockwise_dispatch<F32, B16>(kernelId)(XQ, WQ, x_scale, w_scale, Y, KBatch);
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gen_instances.py`**
```
torch::Tensor &Y,
int KBatch = 1
return gemm_a8w8_bpreshuffle_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, KBatch);
torch::Tensor &Y,
```

**`csrc/ck_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle.h`**
```
torch::Tensor& Y,
int splitK);
```
