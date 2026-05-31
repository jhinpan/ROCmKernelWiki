# Diff summary

- **files changed:** 16
- **lines:** +221 / -18
- **kernel-ish files:** 6

## Files (by churn)

- `Tensile/ReplacementKernels.py`  (+90/-0)
- `Tensile/Tests/unit/test_ReplacementKernels.py`  (+75/-0)
- `Tensile/KernelWriter.py`  (+33/-12)
- `Tensile/TensileCreateLibrary.py`  (+11/-4)
- `Tensile/KernelWriterSource.py`  (+1/-1)
- `tox.ini`  (+1/-1)
- `Tensile/Common.py`  (+1/-0)
- `Tensile/Tests/unit/replacement/bad_file/bad.txt`  (+1/-0)
- `Tensile/Tests/unit/replacement/duplicate_kernel/a.txt`  (+1/-0)
- `Tensile/Tests/unit/replacement/duplicate_kernel/b.txt`  (+1/-0)
- `Tensile/Tests/unit/replacement/known_kernels_v2/baz.s.txt`  (+1/-0)
- `Tensile/Tests/unit/replacement/known_kernels_v2/kernel_named_bar.txt`  (+1/-0)
- `Tensile/Tests/unit/replacement/known_kernels_v2/kernel_named_foo.txt`  (+1/-0)
- `Tensile/Tests/unit/replacement/known_kernels_v3/baz.s.txt`  (+1/-0)
- `Tensile/Tests/unit/replacement/known_kernels_v3/kernel_named_bar.txt`  (+1/-0)

## Key added lines (kernel files)

**`Tensile/Common.py`**
```
globalParameters["MaxFileName"] = 128 # If a file name would be longer than this, shorten it with a hash.
```

**`Tensile/KernelWriter.py`**
```
from .Common import globalParameters, CHeader, roundUp
from .ReplacementKernels import ReplacementKernels
def getKernelFileBase(self, kernel):
rv = self.getKernelName(kernel)
```

**`Tensile/KernelWriterSource.py`**
```
kernelName = self.getKernelFileBase(kernel)
```

**`Tensile/ReplacementKernels.py`**
```
from .Common import globalParameters
import os
class ReplacementKernels:
def __init__(self, dirpath, codeObjectVersion):
```

**`Tensile/TensileCreateLibrary.py`**
```
kernelName = kernelWriter.getKernelFileBase(kernel)
objectFiles = list([kernelWriterAssembly.getKernelFileBase(k) + '.o' \
assemblyKernelNames = [kernelWriterAssembly.getKernelFileBase(k) for k in archKernels]
if len(src.strip()) == 0:
```
