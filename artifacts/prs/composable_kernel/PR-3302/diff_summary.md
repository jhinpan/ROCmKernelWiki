# Diff summary

- **files changed:** 40
- **lines:** +408 / -78
- **kernel-ish files:** 37

## Files (by churn)

- `script/update_amd_copyright_headers.py`  (+295/-0)
- `script/check_copyright_year.sh`  (+61/-9)
- `.pre-commit-config.yaml`  (+6/-6)
- `include/ck_tile/remod.py`  (+2/-3)
- `test/ck_tile/core/arch/mma/test_amdgcn_mma.cpp`  (+2/-2)
- `test/ck_tile/core/arch/test_arch.cpp`  (+2/-2)
- `.github/scripts/therock_configure_ci.py`  (+3/-0)
- `include/ck_tile/core.hpp`  (+1/-2)
- `include/ck_tile/host.hpp`  (+1/-2)
- `include/ck_tile/ops/add_rmsnorm2d_rdquant.hpp`  (+1/-2)
- `include/ck_tile/ops/batched_contraction.hpp`  (+1/-2)
- `include/ck_tile/ops/batched_transpose.hpp`  (+1/-2)
- `include/ck_tile/ops/common.hpp`  (+1/-2)
- `include/ck_tile/ops/elementwise.hpp`  (+1/-2)
- `include/ck_tile/ops/epilogue.hpp`  (+1/-2)

## Key added lines (kernel files)

**`include/ck_tile/remod.py`**
```
HEADER_COMMON = """// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
```

**`script/update_amd_copyright_headers.py`**
```
Normalize and enforce AMD two-line copyright + SPDX headers across files.
Target files:
- C/C++-style: .cpp, .hpp, .inc  -> uses "//" comment style
- Hash-style:  .py, .cmake, .sh, and CMakeLists.txt -> uses "#" style
```
