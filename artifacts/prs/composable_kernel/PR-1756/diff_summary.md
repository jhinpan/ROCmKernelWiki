# Diff summary

- **files changed:** 17
- **lines:** +342 / -367
- **kernel-ish files:** 17

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+68/-187)
- `include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`  (+115/-28)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+39/-36)
- `include/ck_tile/core/arch/arch.hpp`  (+51/-6)
- `example/ck_tile/17_grouped_gemm/utils.hpp`  (+0/-38)
- `include/ck_tile/core/utility/amd_address_space.hpp`  (+0/-37)
- `include/ck_tile/host/host_tensor.hpp`  (+34/-1)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+10/-10)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+6/-3)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+4/-4)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+4/-4)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+4/-4)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+4/-4)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+1/-2)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
using TilePartitioner = ck_tile::GemmTile2DPartitioner<CodegenGemmShape>;
TilePartitioner::MPerBlock,
TilePartitioner::NPerBlock>>,
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
using TilePartitioner = ck_tile::GemmTile2DPartitioner<GemmShape>;
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
using TilePartitioner = ck_tile::GemmTile2DPartitioner<CodegenGemmShape>;
TilePartitioner::MPerBlock,
TilePartitioner::NPerBlock>>,
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`**
```
std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs);
float grouped_gemm(const std::vector<grouped_gemm_kargs>& gemm_descs,
const ck_tile::stream_config& s,
void* p_workspace_);
```
