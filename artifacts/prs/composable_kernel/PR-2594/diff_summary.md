# Diff summary

- **files changed:** 113
- **lines:** +610 / -531
- **kernel-ish files:** 111

## Files (by churn)

- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`  (+81/-81)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+29/-9)
- `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`  (+25/-10)
- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+12/-12)
- `example/ck_tile/05_reduce/reduce.cpp`  (+12/-12)
- `example/ck_tile/06_permute/permute.cpp`  (+12/-12)
- `example/ck_tile/13_moe_sorting/moe_sorting_api.cpp`  (+12/-12)
- `test/ck_tile/elementwise/test_elementwise_1d.cpp`  (+11/-13)
- `test/ck_tile/moe_sorting/moe_sorting_api.cpp`  (+12/-12)
- `test/ck_tile/reduce/test_reduce2d.cpp`  (+12/-12)
- `example/ck_tile/21_elementwise/elementwise_example.cpp`  (+11/-11)
- `example/ck_tile/21_elementwise/elementwise_example_transpose.cpp`  (+11/-11)
- `example/ck_tile/21_elementwise/elementwise_example_unary.cpp`  (+11/-11)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_util.hpp`  (+6/-16)
- `test/ck_tile/memory_copy/test_copy.cpp`  (+11/-11)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py`**
```
const dim3 blocks                      = k_::BlockSize();
return ck_tile::launch_kernel(s, ck_tile::make_kernel<kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs));
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
const dim3 blocks                      = k_::BlockSize();
s, ck_tile::make_kernel<kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs));
const dim3 blocks                      = k_::BlockSize();
ck_tile::make_kernel<kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs)(
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
const dim3 blocks                      = k_::BlockSize();
return ck_tile::launch_kernel(s, ck_tile::make_kernel<kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs));
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_appendkv.py`**
```
const dim3 blocks                      = k_::BlockSize();
return ck_tile::launch_kernel(s, ck_tile::make_kernel<kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs));
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
const dim3 blocks                      = k_::BlockSize();
ck_tile::make_kernel<kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs)(ck_tile::stream_config{{s.stream_id_}});
const dim3 blocks                      = k_::BlockSize();
ck_tile::make_kernel<kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs)(ck_tile::stream_config{{s.stream_id_}});
```
