# Diff summary

- **files changed:** 39
- **lines:** +457 / -231
- **kernel-ish files:** 10

## Files (by churn)

- `Tensile/KernelWriterAssembly.py`  (+303/-150)
- `Tensile/Common.py`  (+37/-34)
- `Tensile/Code.py`  (+35/-1)
- `Tensile/AsmMemoryInstruction.py`  (+7/-7)
- `Tensile/KernelWriter.py`  (+6/-6)
- `Tensile/Source/lib/source/ocl/OclUtils.cpp`  (+12/-0)
- `Tensile/cmake/TensileConfig.cmake`  (+8/-2)
- `Tensile/Source/lib/source/AMDGPU.cpp`  (+6/-0)
- `pytest.ini`  (+6/-0)
- `Tensile/Source/lib/include/Tensile/AMDGPU.hpp`  (+4/-1)
- `Tensile/Source/CMakeLists.txt`  (+2/-2)
- `Tensile/Tests/pre_checkin/mfma/hpa_igemm_i8_split_lds.yaml`  (+2/-2)
- `Tensile/Source/lib/include/Tensile/Serialization/Predicates.hpp`  (+3/-0)
- `Tensile/Components/MAC_F16_HPA.py`  (+1/-1)
- `Tensile/Tests/pre_checkin/denorm/mfma/bfloat16_1k_denorm.yaml`  (+1/-1)

## Key added lines (kernel files)

**`Tensile/AsmMemoryInstruction.py`**
```
if (name == "_ds_load_b128"):
elif (name == "_ds_store_b128"):
elif (name == "_ds_store2_b64"):
elif (name == "_ds_store_b64"):
```

**`Tensile/Code.py`**
```
class SrdUpperFields11XX(BitfieldStructure):
_fields_ = [("dst_sel_x",      ctypes.c_uint, 3),
("dst_sel_y",      ctypes.c_uint, 3),
("dst_sel_z",      ctypes.c_uint, 3),
```

**`Tensile/Common.py`**
```
globalParameters["SupportedISA"] = [(8,0,3), (9,0,0), (9,0,6), (9,0,8), (9,0,10), (10,1,0), (10,1,1), (10,1,2), (10,3,0)
'gfx1010':'navi10', 'gfx1011':'navi12', 'gfx1012':'navi14', 'gfx1030':'navi21',
'gfx1100':'navi31', 'gfx1101':'navi32', 'gfx1102':'navi33'
rv["SupportedISA"]      = tryAssembler(isaVersion, "")
```

**`Tensile/Components/MAC_F16_HPA.py`**
```
vars["instruction"] = "_v_dot2acc_f32_f16"
```

**`Tensile/KernelWriter.py`**
```
readsInc += str(itemGR).count("_buffer_load")
count += globalReadStr.count("_buffer_load")
count += localWriteStr.count("_buffer_load")
numGlobalStoreCinTemplate  = tmpStr.count("_buffer_store")  # count _buffer_store
```
