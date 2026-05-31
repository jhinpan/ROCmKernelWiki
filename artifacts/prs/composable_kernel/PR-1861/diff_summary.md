# Diff summary

- **files changed:** 15
- **lines:** +492 / -135
- **kernel-ish files:** 14

## Files (by churn)

- `example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.cpp`  (+138/-23)
- `example/ck_tile/10_rmsnorm2d/generate.py`  (+94/-62)
- `include/ck_tile/ops/epilogue/default_2d_and_dynamic_quant_epilogue.hpp`  (+91/-0)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_two_pass.hpp`  (+42/-23)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_two_pass.hpp`  (+33/-14)
- `include/ck_tile/ops/rmsnorm2d/kernel/rmsnorm2d_fwd_kernel.hpp`  (+32/-2)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_one_pass.hpp`  (+22/-4)
- `include/ck_tile/host/reference/reference_rmsnorm2d_fwd.hpp`  (+11/-2)
- `example/ck_tile/10_rmsnorm2d/example_rmsnorm2d_fwd.cpp`  (+9/-2)
- `example/ck_tile/10_rmsnorm2d/script/smoke_test.sh`  (+10/-1)
- `example/ck_tile/02_layernorm2d/generate.py`  (+2/-2)
- `example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.hpp`  (+3/-0)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_problem.hpp`  (+2/-0)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_traits.hpp`  (+2/-0)
- `include/ck_tile/ops/epilogue.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/02_layernorm2d/generate.py`**
```
'big'  :[ h_traits('x', 'y', 'xs', 'ys', 1,  1,  1,1024, 8,  True,  False, True, True,    True,   0,    0,    0),
h_traits('x', 'y', 'xs', 'ys', 1, 12,  1, 256, 2,  True,  False, True, True,    True,   0,    0,    0),
```

**`example/ck_tile/10_rmsnorm2d/example_rmsnorm2d_fwd.cpp`**
```
using UnquantYDataType    = ck_tile::null_type;
ck_tile::HostTensor<UnquantYDataType> unquant_y_host_ref({m, n}, {stride, 1});
false, // kSaveUnquant
UnquantYDataType,
```

**`example/ck_tile/10_rmsnorm2d/generate.py`**
```
typename UnquantYDataType_,
bool kSaveUnquant_,
using UnquantYDataType    = ck_tile::remove_cvref_t<UnquantYDataType_>;
static constexpr bool kPadN        = kPadN_;
```

**`example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.cpp`**
```
.insert("save_unquant", "0", "save result before quant")
bool SaveRms,
bool SaveUnquant>
if((fused_quant == 0) && SaveUnquant)
```

**`example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.hpp`**
```
using UnquantYDataType    = ck_tile::half_t;
using UnquantYDataType    = ck_tile::bf16_t;
bool save_unquant;
```
