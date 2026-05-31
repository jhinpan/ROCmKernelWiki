# Diff summary

- **files changed:** 34
- **lines:** +2267 / -285
- **kernel-ish files:** 27

## Files (by churn)

- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_util.hpp`  (+407/-0)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+274/-111)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases.inc`  (+334/-0)
- `example/ck_tile/19_gemm_multi_d/gemm_multi_d_fp16.cpp`  (+296/-0)
- `example/ck_tile/19_gemm_multi_d/run_gemm_multi_d_fp16_example.inc`  (+247/-0)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+59/-42)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+84/-17)
- `example/ck_tile/19_gemm_multi_d/gemm_multi_d_fp16.hpp`  (+79/-0)
- `example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`  (+47/-21)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+34/-28)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+44/-14)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+52/-0)
- `example/ck_tile/19_gemm_multi_d/utils.hpp`  (+50/-0)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+28/-16)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d.cpp`  (+39/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
typename DsDataType,
typename DsLayout,
bool Persistent,
typename CDEElementWise = ck_tile::element_wise::PassThrough>
```

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
typename DsDataType,
typename DsLayout,
bool Persistent = false,
typename CDEElementWise>
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
typename DsDataType,
typename DsLayout,
typename CLayout,
typename CDEElementWise = ck_tile::element_wise::PassThrough>
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
typename DsDataType,
typename DsLayout,
typename ELayout,
bool Persistent,
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
template <typename ADataType,
typename BDataType,
typename DsDataType,
typename AccDataType,
```
