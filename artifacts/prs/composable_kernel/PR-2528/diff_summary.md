# Diff summary

- **files changed:** 51
- **lines:** +1794 / -1212
- **kernel-ish files:** 44

## Files (by churn)

- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+333/-245)
- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+235/-157)
- `example/ck_tile/01_fmha/codegen/ops/fmha_pagedkv_prefill.py`  (+193/-126)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+210/-94)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_appendkv.py`  (+157/-103)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+44/-94)
- `Jenkinsfile`  (+50/-26)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_appendkv_kernel.hpp`  (+37/-37)
- `include/ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp`  (+27/-46)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp`  (+33/-31)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline_default_policy.hpp`  (+34/-30)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs_default_policy.hpp`  (+17/-43)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_whole_k_prefetch_default_policy.hpp`  (+17/-43)
- `example/ck_tile/01_fmha/codegen/utils.py`  (+50/-0)
- `example/ck_tile/01_fmha/codegen/arch.py`  (+42/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/arch.py`**
```
from dataclasses import dataclass, field
from typing import Any, List, Callable
@dataclass(frozen=True)
class ArchTrait:
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py`**
```
from codegen.utils import update_file
per_dtypes += "    (void)t; (void)s; (void)a;"
update_file(autogen_dir / kernel.filename, kernel.template)
update_file(autogen_dir / FMHA_FWD_API_FILENAME, api_pool.api)
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
from collections import OrderedDict
from dataclasses import dataclass
from codegen.arch import ArchTrait, get_factories_for_targets
from codegen.utils import check_duplicates_and_paddings, if_, indent, update_file
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
from collections import OrderedDict
from dataclasses import dataclass, field
from codegen.arch import ArchTrait, get_factories_for_targets
from codegen.utils import check_duplicates_and_paddings, if_, indent, update_file
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_appendkv.py`**
```
import itertools
from collections import OrderedDict
from dataclasses import dataclass
from codegen.arch import ArchTrait, get_factories_for_targets
```
