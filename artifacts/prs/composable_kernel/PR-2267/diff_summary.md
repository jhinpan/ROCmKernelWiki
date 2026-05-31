# Diff summary

- **files changed:** 24
- **lines:** +819 / -305
- **kernel-ish files:** 19

## Files (by churn)

- `tile_engine/ops/gemm/gemm_instance_builder.py`  (+277/-131)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`  (+179/-2)
- `tile_engine/ops/gemm/codegen_utils.py`  (+120/-48)
- `tile_engine/ops/gemm/json_config.py`  (+58/-49)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+46/-7)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+10/-28)
- `test/ck_tile/gemm/test_gemm_pipeline_kernel_types.hpp`  (+20/-11)
- `tile_engine/ops/gemm/gemm_profiler.hpp`  (+15/-11)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+17/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`  (+14/-3)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+15/-0)
- `tile_engine/ops/gemm/configs/user_provided_config.json`  (+7/-7)
- `tile_engine/ops/gemm/gemm_host_api.hpp`  (+12/-0)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+7/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+6/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
else if(data_type == "i8")
return run_gemm_example_prec_type<ck_tile::int8_t, ck_tile::int8_t, int32_t>(
a_layout, b_layout, argc, argv);
```

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile()
constexpr bool is_8bit_float =
std::is_same_v<PrecType, ck_tile::fp8_t> || std::is_same_v<PrecType, ck_tile::bf8_t>;
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);
ADataType* d_A = static_cast<ADataType*>(a_m_k_dev_buf.GetDeviceBuffer());
BDataType* d_B = static_cast<BDataType*>(b_k_n_dev_buf.GetDeviceBuffer());
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
else if(data_type == "int8")
return run_gemm_example_prec_type<GemmConfig<ck_tile::int8_t>,
ck_tile::int8_t,
ck_tile::int8_t,
```

**`include/ck_tile/core/numeric/integer.hpp`**
```
using int32_t      = int32_t;
```
