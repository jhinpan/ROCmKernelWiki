# Diff summary

- **files changed:** 16
- **lines:** +1527 / -216
- **kernel-ish files:** 12

## Files (by churn)

- `test/ck_tile/grouped_gemm_multi_d/test_grouped_gemm_multi_d_util.hpp`  (+431/-0)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_multi_d_example.inc`  (+389/-0)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_multi_d.hpp`  (+220/-0)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_multi_d.cpp`  (+180/-0)
- `example/ck_tile/17_grouped_gemm/README.md`  (+16/-145)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+68/-40)
- `test/ck_tile/grouped_gemm_multi_d/test_grouped_gemm_multi_d_ut_cases.inc`  (+91/-0)
- `test/ck_tile/grouped_gemm_multi_d/test_grouped_gemm_multi_d.cpp`  (+73/-0)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_util.hpp`  (+16/-6)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+14/-4)
- `test/ck_tile/grouped_gemm_preshuffle/test_grouped_gemm_preshuffle_util.hpp`  (+14/-4)
- `test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_util.hpp`  (+0/-13)
- `test/ck_tile/grouped_gemm_multi_d/CMakeLists.txt`  (+9/-0)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+2/-3)
- `example/ck_tile/17_grouped_gemm/CMakeLists.txt`  (+3/-1)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`**
```
using grouped_gemm_kargs = ck_tile::GroupedGemmHostArgs<>;
return gemm_descs.size() * sizeof(ck_tile::GemmTransKernelArg<>);
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm_multi_d.cpp`**
```
template <typename GemmConfig,
typename ADataType,
typename BDataType,
typename DsDataType,
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm_multi_d.hpp`**
```
using ADataType   = ck_tile::half_t;
using BDataType   = ck_tile::half_t;
using D0DataType  = ck_tile::half_t;
using D1DataType  = ck_tile::half_t;
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`**
```
std::vector<ck_tile::GemmTransKernelArg<>> kargs;
kargs.size() * sizeof(ck_tile::GemmTransKernelArg<>),
gemm_descs.push_back({p_a,
{/*ds_ptr*/},
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_multi_d_example.inc`**
```
struct MultiplyMultiply
template <typename E, typename C, typename D0, typename D1>
CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const D0& d0, const D1& d1) const -> void
const float x0_f = ck_tile::type_convert<float>(c) * ck_tile::type_convert<float>(d0) *
```
