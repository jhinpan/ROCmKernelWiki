# Diff summary

- **files changed:** 17
- **lines:** +913 / -690
- **kernel-ish files:** 11

## Files (by churn)

- `python/perf-kernels/tools/plot-layout/plot_layout.py`  (+128/-609)
- `python/perf-kernels/tools/plot-layout/dot/plot_dot.py`  (+287/-0)
- `python/perf-kernels/tools/plot-layout/lds/plot_lds.py`  (+232/-0)
- `python/perf-kernels/tools/plot-layout/README.md`  (+112/-81)
- `python/perf-kernels/tools/plot-layout/blocked/plot_blocked.py`  (+74/-0)
- `python/perf-kernels/tools/plot-layout/utils/utils.py`  (+33/-0)
- `python/perf-kernels/tools/plot-layout/wmma/plot_wmma.py`  (+32/-0)
- `python/perf-kernels/tools/plot-layout/blocked/__init__.py`  (+3/-0)
- `python/perf-kernels/tools/plot-layout/dot/__init__.py`  (+3/-0)
- `python/perf-kernels/tools/plot-layout/lds/__init__.py`  (+3/-0)
- `python/perf-kernels/tools/plot-layout/utils/__init__.py`  (+3/-0)
- `python/perf-kernels/tools/plot-layout/wmma/__init__.py`  (+3/-0)
- `python/perf-kernels/tools/plot-layout/blocked/blockedLayout.tex`  (+0/-0)
- `python/perf-kernels/tools/plot-layout/dot/dotLayout.tex`  (+0/-0)
- `python/perf-kernels/tools/plot-layout/lds/ldsLayout.tex`  (+0/-0)

## Key added lines (kernel files)

**`python/perf-kernels/tools/plot-layout/blocked/__init__.py`**
```
from .plot_blocked import generate_blocked_tex
__all__ = ["generate_blocked_tex"]
```

**`python/perf-kernels/tools/plot-layout/blocked/plot_blocked.py`**
```
from dataclasses import dataclass
from pathlib import Path
@dataclass
class BlockedConfig:
```

**`python/perf-kernels/tools/plot-layout/dot/__init__.py`**
```
from .plot_dot import generate_dot_tex
__all__ = ["generate_dot_tex"]
```

**`python/perf-kernels/tools/plot-layout/dot/plot_dot.py`**
```
from dataclasses import dataclass
from pathlib import Path
@dataclass
class DotConfig:
```

**`python/perf-kernels/tools/plot-layout/lds/__init__.py`**
```
from .plot_lds import generate_lds_tex
__all__ = ["generate_lds_tex"]
```
