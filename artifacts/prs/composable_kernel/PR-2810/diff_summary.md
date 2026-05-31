# Diff summary

- **files changed:** 11
- **lines:** +39 / -23
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/ops/reduce/pipeline/reduce2d_shape.hpp`  (+12/-5)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`  (+6/-2)
- `example/ck_tile/06_permute/alternative_impl/matrix_core_swizzle_kernel.hpp`  (+3/-4)
- `example/ck_tile/05_reduce/reduce.cpp`  (+2/-3)
- `include/ck_tile/ops/elementwise/kernel/elementwise_kernel.hpp`  (+4/-1)
- `example/ck_tile/06_permute/permute.cpp`  (+4/-0)
- `include/ck_tile/ops/reduce/kernel/reduce2d_kernel.hpp`  (+4/-0)
- `example/ck_tile/21_elementwise/elementwise_example.cpp`  (+1/-2)
- `example/ck_tile/21_elementwise/elementwise_example_add_4d.cpp`  (+1/-2)
- `example/ck_tile/21_elementwise/elementwise_example_transpose.cpp`  (+1/-2)
- `example/ck_tile/21_elementwise/elementwise_example_unary.cpp`  (+1/-2)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`**
```
using Kernel                      = ck_tile::Reduce<Problem>;
const ck_tile::index_t kBlockSize = Kernel::BlockSize();
return !run_gemm_example<GemmConfigComputeV3_WMMA>(arg_parser);
```

**`example/ck_tile/05_reduce/reduce.cpp`**
```
using Kernel                      = ck_tile::Reduce<Porblem>;
const ck_tile::index_t kBlockSize = Kernel::BlockSize();
```

**`example/ck_tile/06_permute/alternative_impl/matrix_core_swizzle_kernel.hpp`**
```
static constexpr int BLOCK_SIZE                   = BLOCK_SIZE_;
static constexpr int WavesPerBlock_N              = BLOCK_SIZE / ck_tile::get_warp_size();
static constexpr int WavesPerBlock_K              = 1;
```

**`example/ck_tile/06_permute/permute.cpp`**
```
.insert("json", "0", "0: No Json, 1: Dump Results in Json format")
```

**`example/ck_tile/21_elementwise/elementwise_example.cpp`**
```
const ck_tile::index_t kBlockSize = Kernel::BlockSize();
```
