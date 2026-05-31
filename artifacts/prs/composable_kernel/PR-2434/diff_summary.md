# Diff summary

- **files changed:** 34
- **lines:** +2738 / -340
- **kernel-ish files:** 31

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v1.hpp`  (+472/-0)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+450/-0)
- `test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_util.hpp`  (+384/-0)
- `include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`  (+251/-108)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`  (+294/-0)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+94/-43)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+108/-16)
- `include/ck_tile/ops/gemm/block/block_wp_asmem_bsmem_creg_v1.hpp`  (+122/-0)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+63/-23)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+59/-24)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+54/-22)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+71/-0)
- `example/ck_tile/18_flatmm/run_flatmm_example.inc`  (+45/-21)
- `example/ck_tile/18_flatmm/flatmm_basic.hpp`  (+30/-31)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+24/-15)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile_flatmm()
if constexpr(M_Warp_Tile == 32)
return sizeof(PrecType) == 2 ? 16 : 64;
```

**`example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`**
```
template <typename GemmConfig,
typename ADataType,
typename BDataType,
typename DsDataType,
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
template <typename GemmConfig, typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t)
assert(t.get_lengths().size() == 2);
int n_                = t.get_lengths()[1];
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
GemmConfig::NumWaveGroups,
GemmConfig::Preshuffle>;
UniversalGemmProblem::kBlockSize,
<< "problem: " << UniversalGemmProblem::GetName() << '\n'
```

**`example/ck_tile/18_flatmm/flatmm_basic.cpp`**
```
template <typename FlatmmConfig,
typename ADataType,
typename DsDatatype,
typename DsLayout,
```
