# Diff summary

- **files changed:** 11
- **lines:** +134 / -76
- **kernel-ish files:** 9

## Files (by churn)

- `include/ck_tile/ops/flatmm/kernel/moe_flatmm_kernel.hpp`  (+36/-20)
- `include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`  (+20/-17)
- `include/ck_tile/host/reference/reference_moe_gemm.hpp`  (+17/-15)
- `example/ck_tile/18_flatmm/mixed_prec/a16w4_moe_flatmm.cpp`  (+28/-3)
- `example/ck_tile/18_flatmm/mixed_prec/run_a16w4_moe_flatmm_example.inc`  (+12/-7)
- `example/ck_tile/18_flatmm/mxgemm/run_mx_flatmm.inc`  (+6/-4)
- `example/ck_tile/18_flatmm/mixed_prec/a16w4_moe_flatmm.hpp`  (+4/-3)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm_instance.cpp.in`  (+4/-2)
- `example/ck_tile/18_flatmm/CMakeLists.txt`  (+3/-2)
- `example/ck_tile/18_flatmm/mixed_prec/run_mixed_prec_flatmm.inc`  (+3/-2)
- `example/ck_tile/18_flatmm/mixed_prec/a16w4_flatmm.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/18_flatmm/mixed_prec/a16w4_flatmm.hpp`**
```
static constexpr ck_tile::index_t M_Tile = 64;
```

**`example/ck_tile/18_flatmm/mixed_prec/a16w4_moe_flatmm.cpp`**
```
std::cout << "Launching kernel " << Kernel::GetName() << "\n"
<< "with args:" << CodegenFlatmmShape::GetName() << "\n"
<< "k_batch: " << kargs.k_batch << std::endl;
else if(gemm_kind == "gemm1_split_k")
```

**`example/ck_tile/18_flatmm/mixed_prec/a16w4_moe_flatmm.hpp`**
```
static constexpr ck_tile::index_t M_Tile = 32;
"Gemm kind in FFN network [gemm1_gate_up | gemm2 | gemm1_split_k] - "
.insert("repeat", "10", "number of iterations to benchmark the kernel.")
.insert("k_batch", "1", "parallism to control splik-k.");
```

**`example/ck_tile/18_flatmm/mixed_prec/run_a16w4_moe_flatmm_example.inc`**
```
using ADataType = PrecActType;
using BDataType = PrecWeightType;
using ADataType = PrecActType;
using BDataType = PrecWeightType;
```

**`example/ck_tile/18_flatmm/mixed_prec/run_mixed_prec_flatmm.inc`**
```
auto scale_b_dev_ptr =
ck_tile::FlatmmScalePointer<DequantGranularityN, DequantGranularityK, ScaleType>{
static_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()), N / DequantGranularityN};
```
