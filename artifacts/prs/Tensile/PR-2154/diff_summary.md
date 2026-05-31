# Diff summary

- **files changed:** 101
- **lines:** +170 / -100
- **kernel-ish files:** 5

## Files (by churn)

- `Tensile/AsmCaps.py`  (+44/-0)
- `Tensile/Source/lib/include/Tensile/AMDGPU.hpp`  (+15/-1)
- `Tensile/Common.py`  (+3/-3)
- `Tensile/Source/lib/include/Tensile/PlaceholderLibrary.hpp`  (+6/-0)
- `Tensile/Source/CMakeLists.txt`  (+2/-2)
- `Tensile/Tests/pre_checkin/preload_kernel_arguments_always_half.yaml`  (+2/-1)
- `pytest.ini`  (+3/-0)
- `CHANGELOG.md`  (+1/-1)
- `Tensile/Source/lib/include/Tensile/Serialization/Predicates.hpp`  (+2/-0)
- `Tensile/Tests/disabled/direct_to_lds/dtl_dgemm.yaml`  (+1/-1)
- `Tensile/Tests/disabled/direct_to_lds/dtl_dgemm_lite.yaml`  (+1/-1)
- `Tensile/Tests/disabled/direct_to_lds/dtl_tsgr_dgemm.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/b8f8gemm_hybrid_b8f8b8s_SR_gfx940.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/b8f8gemm_hybrid_b8f8b8s_gfx940.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/b8f8gemm_hybrid_b8f8hs_gfx940.yaml`  (+1/-1)

## Key added lines (kernel files)

**`Tensile/AsmCaps.py`**
```
(11, 5, 0): {'HasAddLshl': True,
'HasAtomicAdd': True,
'HasDirectToLdsDest': False,
'HasDirectToLdsNoDest': False,
```

**`Tensile/Common.py`**
```
(11,5,0), (11,5,1),
'gfx1150':'strixpoint', 'gfx1151':'strixhalo',
isasWithDisabledHWMonitor = ((9,4,1), (9,4,2), (11,0,0), (11,0,1), (11,0,2), (11,5,0), (11,5,1), (12,0,0), (12,0,1))
```

**`Tensile/Source/lib/include/Tensile/AMDGPU.hpp`**
```
gfx1102 = 1102,
gfx1150 = 1150,
gfx1151 = 1151
case AMDGPU::Processor::gfx1150:
```

**`Tensile/Source/lib/include/Tensile/PlaceholderLibrary.hpp`**
```
case LazyLoadingInit::gfx1150:
return "TensileLibrary_*_gfx1150";
case LazyLoadingInit::gfx1151:
return "TensileLibrary_*_gfx1151";
```

**`Tensile/Source/lib/include/Tensile/Serialization/Predicates.hpp`**
```
iot::enumCase(io, value, "gfx1150", AMDGPU::Processor::gfx1150);
iot::enumCase(io, value, "gfx1151", AMDGPU::Processor::gfx1151);
```
