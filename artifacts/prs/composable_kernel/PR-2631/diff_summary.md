# Diff summary

- **files changed:** 9
- **lines:** +142 / -178
- **kernel-ish files:** 8

## Files (by churn)

- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+33/-42)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+32/-39)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+26/-31)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_appendkv_kernel.hpp`  (+19/-38)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`  (+17/-16)
- `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`  (+5/-9)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+6/-0)
- `example/ck_tile/03_gemm/README.md`  (+3/-2)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
using CodegenGemmTraits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
GemmConfig::kPadN,
GemmConfig::kPadK,
CLayout>;
```

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
using MemoryOpSet =
std::integral_constant<ck_tile::memory_operation_enum, ck_tile::memory_operation_enum::set>;
using MemoryOpAtomicAdd = std::integral_constant<ck_tile::memory_operation_enum,
ck_tile::memory_operation_enum::atomic_add>;
```

**`example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`**
```
int run_gemm_example_prec_type(std::string a_layout,
std::string b_layout,
ck_tile::ArgParser& arg_parser)
using Row       = ck_tile::tensor_layout::gemm::RowMajor;
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
template <typename CDataType>
bool do_verify(const ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
const ck_tile::HostTensor<CDataType>& c_m_n_ref,
const ck_tile::tuple<double, double>& rtol_atol,
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
Run(has_hot_loop_, tail_number_, MemoryOpSet{});
Run(has_hot_loop_, tail_number_, MemoryOpAtomicAdd{});
int run_gemm_example_prec_type(std::string a_layout,
std::string b_layout,
```
