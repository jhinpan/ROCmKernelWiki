# Diff summary

- **files changed:** 10
- **lines:** +18 / -232
- **kernel-ish files:** 9

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+0/-104)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+6/-31)
- `include/ck_tile/core/utility/type_traits.hpp`  (+0/-30)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+6/-16)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+3/-13)
- `test/ck_tile/gemm/test_gemm_pipeline_persistent.cpp`  (+0/-16)
- `test/ck_tile/gemm/test_gemm_pipeline_kernel_types.hpp`  (+0/-9)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+2/-4)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+1/-4)
- `test/ck_tile/gemm/CMakeLists.txt`  (+0/-5)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
typename CLayout>
```

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
.insert("init", "0", "0:random, 1:linear, 2:constant(1)");
typename CLayout>
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
int n_repeat)
float ave_time =
gemm_calc<ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout>(
<< " : " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
typename CLayout>
GemmConfig::UseStructuredSparsity>;
const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
```

**`test/ck_tile/gemm/test_gemm_pipeline_util.hpp`**
```
using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
TransposeC>;
const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
```
