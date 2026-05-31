# Diff summary

- **files changed:** 22
- **lines:** +440 / -193
- **kernel-ish files:** 22

## Files (by churn)

- `example/ck_tile/19_gemm_multi_d/gemm_multi_d_fp16.hpp`  (+113/-21)
- `example/ck_tile/16_batched_gemm/batched_gemm.hpp`  (+110/-19)
- `example/ck_tile/19_gemm_multi_d/gemm_multi_d_fp16.cpp`  (+28/-54)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+22/-50)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_data.cpp`  (+22/-7)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight.cpp`  (+22/-7)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_forward.cpp`  (+22/-7)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+23/-4)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_utils.hpp`  (+14/-4)
- `example/ck_tile/19_gemm_multi_d/run_gemm_multi_d_fp16_example.inc`  (+10/-5)
- `example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`  (+9/-5)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage.cpp`  (+8/-3)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage_invoker.hpp`  (+7/-2)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp`  (+4/-1)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+4/-1)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_splitk_two_stage.cpp`**
```
template <template <typename PreType, typename WorkspaceType> typename GemmConfig>
return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t, float>,
return run_gemm_example_prec_type<GemmConfig<ck_tile::bf16_t, float>,
return !run_gemm_example<GemmConfigTwoStage_Wmma>(arg_parser);
```

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_invoker.hpp`**
```
template <typename PrecType_, typename WorkspaceType_>
struct GemmConfigTwoStage_Wmma : public GemmConfigComputeV3_WMMA<PrecType_>
using WorkspaceType = ck_tile::remove_cvref_t<WorkspaceType_>;
const ck_tile::index_t kBlockSize      = ElementwiseKernel::BlockSize();
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
template <typename GemmConfig,
typename ADataType,
constexpr ck_tile::index_t M_Tile = GemmConfig::M_Tile;
constexpr ck_tile::index_t N_Tile = GemmConfig::N_Tile;
```

**`example/ck_tile/16_batched_gemm/batched_gemm.hpp`**
```
struct GemmConfigMemory
static constexpr ck_tile::index_t M_Tile = 128;
static constexpr ck_tile::index_t N_Tile = 32;
static constexpr ck_tile::index_t K_Tile = 64;
```

**`example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`**
```
template <typename GemmConfig,
typename ADataType,
float ave_time = batched_gemm<GemmConfig,
ADataType,
```
