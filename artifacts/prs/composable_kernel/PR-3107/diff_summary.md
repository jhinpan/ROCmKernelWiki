# Diff summary

- **files changed:** 8
- **lines:** +298 / -75
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`  (+199/-13)
- `example/ck_tile/40_streamk_gemm/streamk_gemm_basic.cpp`  (+64/-26)
- `example/ck_tile/40_streamk_gemm/run_gemm_example.inc`  (+15/-19)
- `example/ck_tile/40_streamk_gemm/gemm_utils.hpp`  (+9/-9)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm_tile_partitioner.hpp`  (+3/-4)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm_tile_partitioner_impl.hpp`  (+3/-3)
- `include/ck_tile/host/kernel_launch.hpp`  (+4/-0)
- `example/ck_tile/40_streamk_gemm/README.md`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/40_streamk_gemm/gemm_utils.hpp`**
```
template <typename PrecType, bool Persistent_>
static constexpr ck_tile::index_t M_Tile = 256;
static constexpr ck_tile::index_t N_Tile = 256;
static constexpr ck_tile::index_t K_Tile = 16;
```

**`example/ck_tile/40_streamk_gemm/run_gemm_example.inc`**
```
ck_tile::StreamKReductionStrategy reduction_strategy)
ck_tile::reboot::StreamKHostArgs args{a_m_k_dev_buf.GetDeviceBuffer(),
b_k_n_dev_buf.GetDeviceBuffer(),
c_m_n_dev_buf.GetDeviceBuffer(),
```

**`example/ck_tile/40_streamk_gemm/streamk_gemm_basic.cpp`**
```
std::tuple<float, ck_tile::index_t> gemm(const ck_tile::reboot::StreamKHostArgs& args,
using TilePartitioner =
ck_tile::StreamKTilePartitioner_v2<GemmShape, ReductionStrategy, GemmConfig::Persistent>;
using Kernel = ck_tile::reboot::StreamKKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
```

**`include/ck_tile/host/kernel_launch.hpp`**
```
if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
preprocess();
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`**
```
using ADataType   = typename GemmPipeline::ADataType;
using BDataType   = typename GemmPipeline::BDataType;
using CDataType   = typename EpiloguePipeline::ODataType;
using AccDataType = typename EpiloguePipeline::AccDataType;
```
