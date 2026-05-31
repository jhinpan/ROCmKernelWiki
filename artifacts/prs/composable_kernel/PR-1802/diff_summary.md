# Diff summary

- **files changed:** 58
- **lines:** +1826 / -1048
- **kernel-ish files:** 55

## Files (by churn)

- `example/ck_tile/10_rmsnorm2d/generate.py`  (+681/-0)
- `example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.cpp`  (+280/-54)
- `include/ck_tile/ops/rmsnorm2d/kernel/rmsnorm2d_fwd_kernel.hpp`  (+169/-28)
- `example/ck_tile/10_rmsnorm2d/instances/rmsnorm2d_fwd_api.cpp`  (+0/-146)
- `example/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.hpp`  (+34/-85)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_two_pass.hpp`  (+73/-18)
- `include/ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp`  (+55/-31)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_one_pass.hpp`  (+66/-15)
- `example/ck_tile/10_rmsnorm2d/instances/rmsnorm2d_fwd_instance_common.hpp`  (+0/-65)
- `example/ck_tile/02_layernorm2d/layernorm2d_fwd.hpp`  (+31/-28)
- `example/ck_tile/02_layernorm2d/generate.py`  (+28/-28)
- `example/ck_tile/10_rmsnorm2d/script/smoke_test.sh`  (+29/-25)
- `include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_traits.hpp`  (+54/-0)
- `example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`  (+25/-24)
- `include/ck_tile/ops/smoothquant/pipeline/smoothquant_pipeline_two_pass.hpp`  (+24/-21)

## Key added lines (kernel files)

**`example/ck_tile/02_layernorm2d/generate.py`**
```
typename SmoothScaleDataType_,
using SmoothScaleDataType = ck_tile::remove_cvref_t<SmoothScaleDataType_>;
typename SmoothScaleDataType_,
SmoothScaleDataType_,
```

**`example/ck_tile/02_layernorm2d/layernorm2d_fwd.cpp`**
```
.insert("prec_sm",
typename SmoothScaleDataType,
std::string prec_sm = arg_parser.get_str("prec_sm");
if(prec_sm == "auto")
```

**`example/ck_tile/02_layernorm2d/layernorm2d_fwd.hpp`**
```
template <typename InType,
typename OutType,
typename SmoothSScaleDataType_,
typename YScaleDataType_>
```

**`example/ck_tile/10_rmsnorm2d/example_rmsnorm2d_fwd.cpp`**
```
using XDataType           = DataType;
using YDataType           = DataType;
using GammaDataType       = DataType;
using InvRmsDataType      = ck_tile::null_type;
```

**`example/ck_tile/10_rmsnorm2d/generate.py`**
```
import argparse
from enum import IntEnum
from pathlib import Path
import sys
```
