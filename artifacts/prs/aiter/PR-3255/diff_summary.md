# Diff summary

- **files changed:** 11
- **lines:** +139 / -90
- **kernel-ish files:** 11

## Files (by churn)

- `csrc/include/gemm_dispatch_utils.h`  (+87/-37)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8.cu`  (+9/-7)
- `csrc/ck_gemm_a8w8/gemm_a8w8.cu`  (+9/-7)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu`  (+4/-7)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16.cu`  (+4/-6)
- `csrc/ck_deepgemm/deepgemm.cu`  (+6/-4)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale.cu`  (+4/-6)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_cktile.cu`  (+4/-4)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle.cu`  (+4/-4)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle.cu`  (+4/-4)
- `csrc/cktile_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_cktile.cu`  (+4/-4)

## Key added lines (kernel files)

**`csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8.cu`**
```
using BatchedRowwiseKernel = torch::Tensor (*)(torch::Tensor&,
torch::Tensor&,
torch::Tensor&,
torch::Tensor&,
```

**`csrc/ck_batched_gemm_bf16/batched_gemm_bf16.cu`**
```
using BatchedKernel = torch::Tensor (*)(
torch::Tensor&, torch::Tensor&, torch::Tensor&, std::optional<torch::Tensor>, int);
const int cu_num           = get_device_cu_num();
const std::string_view gfx = get_device_gfx();
```

**`csrc/ck_deepgemm/deepgemm.cu`**
```
using RowwiseKernel = torch::Tensor (*)(torch::Tensor&,
torch::Tensor&,
torch::Tensor&,
torch::Tensor&,
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale.cu`**
```
using BlockwiseKernel = torch::Tensor (*)(
torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, int);
const int cu_num           = get_device_cu_num();
const std::string_view gfx = get_device_gfx();
```

**`csrc/ck_gemm_a8w8/gemm_a8w8.cu`**
```
using RowwiseKernel = torch::Tensor (*)(torch::Tensor&,
torch::Tensor&,
torch::Tensor&,
torch::Tensor&,
```
