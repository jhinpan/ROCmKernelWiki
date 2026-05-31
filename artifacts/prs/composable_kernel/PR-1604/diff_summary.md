# Diff summary

- **files changed:** 61
- **lines:** +1792 / -768
- **kernel-ish files:** 55

## Files (by churn)

- `example/ck_tile/02_layernorm2d/generate.py`  (+670/-0)
- `example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`  (+242/-28)
- `include/ck_tile/ops/layernorm2d/kernel/layernorm2d_fwd_kernel.hpp`  (+161/-30)
- `example/ck_tile/02_layernorm2d/instances/layernorm2d_fwd_api.cpp`  (+0/-155)
- `include/ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp`  (+140/-0)
- `include/ck_tile/core/numeric/int8.hpp`  (+104/-0)
- `example/ck_tile/02_layernorm2d/layernorm2d_fwd.hpp`  (+23/-80)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_one_pass.hpp`  (+68/-14)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_two_pass.hpp`  (+64/-15)
- `example/ck_tile/02_layernorm2d/README.md`  (+66/-3)
- `example/ck_tile/02_layernorm2d/instances/layernorm2d_fwd_instance_common.hpp`  (+0/-67)
- `example/ck_tile/02_layernorm2d/script/perf_test.sh`  (+33/-33)
- `example/ck_tile/02_layernorm2d/script/smoke_test.sh`  (+29/-25)
- `include/ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_traits.hpp`  (+54/-0)
- `include/ck_tile/host/reference/reference_layernorm2d_fwd.hpp`  (+32/-5)

## Key added lines (kernel files)

**`example/ck_tile/02_layernorm2d/generate.py`**
```
import argparse
from enum import IntEnum
from pathlib import Path
import sys
```

**`example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`**
```
.insert("prec_i", "fp16", "input precision")
.insert("prec_o", "auto", "output precision, set auto will be the same as input")
.insert("prec_sx",
"output quant scale type, set auto will use fp32. used when fquant=1")
```

**`example/ck_tile/02_layernorm2d/layernorm2d_fwd.hpp`**
```
template <typename InType, typename OutType, typename XScaleDataType_, typename YScaleDataType_>
template <typename OutType, typename XScaleDataType_, typename YScaleDataType_>
struct LayerNormTypeConfig<ck_tile::half_t, OutType, XScaleDataType_, YScaleDataType_>
using YDataType       = OutType;
```

**`include/ck_tile/core/numeric/int8.hpp`**
```
namespace ck_tile {
using int8_t = int8_t;
template <class T>
struct numeric;
```

**`include/ck_tile/core/numeric/type_convert.hpp`**
```
CK_TILE_TYPE_CONVERT(float, float, int8_t, int8)
CK_TILE_TYPE_CONVERT(int8_t, int8, float, float)
```
