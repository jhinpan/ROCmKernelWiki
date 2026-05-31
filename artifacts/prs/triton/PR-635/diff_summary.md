# Diff summary

- **files changed:** 13
- **lines:** +1594 / -0
- **kernel-ish files:** 8

## Files (by churn)

- `python/perf-kernels/tools/plot-layout/tikzplot.tex`  (+880/-0)
- `python/perf-kernels/tools/plot-layout/plot_layout.py`  (+290/-0)
- `python/perf-kernels/tools/amdgcn-cfg/amdgcn-cfg.py`  (+222/-0)
- `python/perf-kernels/tools/plot-layout/README.md`  (+117/-0)
- `python/perf-kernels/tools/occ.sh`  (+71/-0)
- `python/perf-kernels/tools/amdgcn-cfg/README.md`  (+14/-0)
- `python/perf-kernels/tools/tune_gemm/README.md`  (+0/-0)
- `python/perf-kernels/tools/tune_gemm/icache_flush.py`  (+0/-0)
- `python/perf-kernels/tools/tune_gemm/matmul_kernel.py`  (+0/-0)
- `python/perf-kernels/tools/tune_gemm/one_config.py`  (+0/-0)
- `python/perf-kernels/tools/tune_gemm/tune_gemm.py`  (+0/-0)
- `python/perf-kernels/tools/tune_gemm/utils/file_generator.py`  (+0/-0)
- `python/perf-kernels/tools/tune_gemm/utils/utils.py`  (+0/-0)

## Key added lines (kernel files)

**`python/perf-kernels/tools/amdgcn-cfg/amdgcn-cfg.py`**
```
import os
import argparse
import re
from collections import OrderedDict
```

**`python/perf-kernels/tools/plot-layout/plot_layout.py`**
```
import argparse
import sys
import os
import subprocess
```
