# Diff summary

- **files changed:** 13
- **lines:** +863 / -369
- **kernel-ish files:** 13

## Files (by churn)

- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+253/-101)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+250/-53)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_util.hpp`  (+135/-83)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+128/-64)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+12/-33)
- `example/ck_tile/16_batched_gemm/batched_gemm.hpp`  (+32/-8)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+26/-2)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+14/-14)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+6/-3)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_ut_cases.inc`  (+3/-3)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+2/-2)
- `test/ck_tile/batched_gemm/test_batched_gemm_ut_cases.inc`  (+2/-2)
- `example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`  (+0/-1)

## Key added lines (kernel files)

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
constexpr ck_tile::index_t M_Tile = 128;
constexpr ck_tile::index_t N_Tile = 32;
constexpr ck_tile::index_t K_Tile = 64;
constexpr ck_tile::index_t M_Warp = 4;
```

**`example/ck_tile/16_batched_gemm/batched_gemm.hpp`**
```
arg_parser.insert("m", "512", "m dimension")
.insert("n", "1024", "n dimension")
.insert("k", "2048", "k dimension")
.insert("batch_stride_a", "1048576", "Batch A stride")
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
return gemm_descs.size() * sizeof(ck_tile::GemmTransKernelArg);
constexpr ck_tile::index_t M_Tile = 128;
constexpr ck_tile::index_t N_Tile = 32;
constexpr ck_tile::index_t K_Tile = 64;
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`**
```
using grouped_gemm_kargs = ck_tile::GemmHostArgs;
.insert("group_count", "8", "group count.");
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`**
```
Ns.push_back(256 + 512 * i);
Ks.push_back(256 + 64 * i);
static constexpr ck_tile::index_t k_batch = 1;
gemm_descs.push_back(
```
