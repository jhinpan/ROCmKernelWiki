# Diff summary

- **files changed:** 15
- **lines:** +330 / -145
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`  (+139/-65)
- `include/ck_tile/host/permute_pk_int4.hpp`  (+78/-0)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+3/-56)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+31/-4)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+18/-0)
- `example/ck_tile/38_block_scale_gemm/run_gemm_bquant_example.inc`  (+12/-1)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+12/-1)
- `test/ck_tile/gemm_block_scale/test_run_gemm_aquant_example.inc`  (+12/-1)
- `example/ck_tile/38_block_scale_gemm/run_gemm_aquant_example.inc`  (+11/-1)
- `include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+4/-6)
- `include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`  (+2/-5)
- `example/ck_tile/03_gemm/script/smoke_test_mem_pipeline.sh`  (+4/-1)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+2/-2)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_basic.cpp`  (+1/-1)
- `include/ck_tile/core/numeric/pk_int4.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
template <>
struct GemmTypeConfig<ck_tile::fp8_t, ck_tile::pk_int4_t, ck_tile::half_t>
using ADataType   = ck_tile::fp8_t;
using BDataType   = ck_tile::pk_int4_t;
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
else if(data_type == "fp16i4")
else if(data_type == "fp8i4")
if constexpr(GemmConfig<ck_tile::fp8_t>::Pipeline == CK_TILE_PIPELINE_COMPUTE_V3)
return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_basic.cpp`**
```
int main(int argc, char* argv[]) { return !run_gemm_example<GemmConfigQuant>(argc, argv); }
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_aquant_example.inc`**
```
if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
ck_tile::HostTensor<ADataType> a_m_k_dev = a_m_k;
ck_tile::permute_vectors_i4x4_b(a_m_k_dev);
a_m_k_dev_buf.ToDevice(a_m_k_dev.data());
```
