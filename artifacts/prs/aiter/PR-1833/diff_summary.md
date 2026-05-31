# Diff summary

- **files changed:** 15
- **lines:** +1261 / -0
- **kernel-ish files:** 13

## Files (by churn)

- `aiter/ops/triton/utils/_triton/tunning/screen.py`  (+363/-0)
- `aiter/ops/triton/utils/_triton/tunning/view-screen.py`  (+172/-0)
- `aiter/ops/triton/utils/_triton/tunning/_utils.py`  (+150/-0)
- `aiter/ops/triton/utils/_triton/tunning/verify-perf.py`  (+81/-0)
- `aiter/ops/triton/utils/_triton/tunning/README.md`  (+71/-0)
- `aiter/ops/triton/utils/_triton/tunning/rprof.py`  (+69/-0)
- `aiter/ops/triton/utils/_triton/tunning/ut_a8w8_gemm_blockscale_preshuffle.py`  (+50/-0)
- `aiter/ops/triton/utils/_triton/tunning/ut_a8w8_gemm_blockscale.py`  (+48/-0)
- `aiter/ops/triton/utils/_triton/tunning/ut_a16w8_gemm_blockscale_preshuffle.py`  (+47/-0)
- `aiter/ops/triton/utils/_triton/tunning/ut_afp4wfp4_gemm.py`  (+44/-0)
- `aiter/ops/triton/utils/_triton/tunning/ut_afp4wfp4_gemm_preshuffle.py`  (+44/-0)
- `aiter/ops/triton/utils/_triton/tunning/ut_a16w16_gemm.py`  (+42/-0)
- `aiter/ops/triton/utils/_triton/tunning/ut_a8w8_gemm_per_token_scale.py`  (+40/-0)
- `aiter/ops/triton/utils/_triton/tunning/ut_template.py`  (+36/-0)
- `aiter/ops/triton/utils/_triton/tunning/.gitignore`  (+4/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/utils/_triton/tunning/_utils.py`**
```
import os
import triton
from triton.testing import runtime
import torch
```

**`aiter/ops/triton/utils/_triton/tunning/rprof.py`**
```
import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="")
```

**`aiter/ops/triton/utils/_triton/tunning/screen.py`**
```
from itertools import product
import os
import sys
import triton
```

**`aiter/ops/triton/utils/_triton/tunning/ut_a16w16_gemm.py`**
```
import sys
from _utils import (
run_profile,
get_input_shape_and_config_list,
```

**`aiter/ops/triton/utils/_triton/tunning/ut_a16w8_gemm_blockscale_preshuffle.py`**
```
import sys
from _utils import (
run_profile,
get_input_shape_and_config_list,
```
