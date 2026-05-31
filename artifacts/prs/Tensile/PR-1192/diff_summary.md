# Diff summary

- **files changed:** 35
- **lines:** +97 / -92
- **kernel-ish files:** 6

## Files (by churn)

- `Tensile/TensileCreateLibrary.py`  (+39/-15)
- `.jenkins/common.groovy`  (+10/-10)
- `Tensile/Source/client/CMakeLists.txt`  (+0/-13)
- `Tensile/Source/lib/CMakeLists.txt`  (+0/-13)
- `Tensile/Source/client/source/Reference.cpp`  (+9/-1)
- `Tensile/Tests/unit/test_tryAssembler.py`  (+6/-4)
- `Tensile/KernelWriterAssembly.py`  (+2/-4)
- `Tensile/Source/CMakeLists.txt`  (+3/-3)
- `Tensile/Components/Signature.py`  (+2/-3)
- `Tensile/Common.py`  (+1/-1)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_BBH_MT32x32x32_AF0EM8_ASEM8_FL0_GRVW2_ISA908_MDA2_PGR1_PLR1_SU32_TT2_2_VAW1_VW2_WG16_16_1_WGM8.s.txt`  (+1/-1)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_BBH_MT32x32x32_AF0EM8_ASEM8_FL0_GRVW2_ISA908_PGR1_PLR1_SU32_TT2_2_VAW1_VW2_WG16_16_1_WGM8.s.txt`  (+1/-1)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_BBH_MT32x32x32_SE_K1.s.txt`  (+1/-1)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_BBH_MT64x128x64_AF0EM8_ASEM8_FL0_GRVW4_ISA908_MDA2_PGR1_PLR1_SU32_TT4_4_VAW1_VW4_WG16_32_1_WGM8.s.txt`  (+1/-1)
- `Tensile/ReplacementKernels-cov3/Cijk_Alik_Bljk_BBH_MT64x128x64_AF0EM8_ASEM8_FL0_GRVW4_ISA908_PGR1_PLR1_SU32_TT4_4_VAW1_VW4_WG16_32_1_WGM8.s.txt`  (+1/-1)

## Key added lines (kernel files)

**`Tensile/Common.py`**
```
rv["HasCodeObjectV3"] = tryAssembler(isaVersion, "", False, "-mllvm --amdhsa-code-object-version=2")
```

**`Tensile/Components/Signature.py`**
```
kStr += ".amdgcn_target \"amdgcn-amd-amdhsa--gfx%s\"%s" \
% ("".join(map(str,writer.version)), writer.endLine)
```

**`Tensile/KernelWriterAssembly.py`**
```
rv += ['-mllvm --amdhsa-code-object-version=2' if globalParameters["CodeObjectVersion"] == "V2" else '-mllvm --amdhsa-co
rv += ['-mcpu=gfx' + ''.join(map(str,isa))]
```

**`Tensile/Source/client/source/Reference.cpp`**
```
void throwException(const std::string& msg)
throw std::runtime_error(msg.c_str());
throwException(msg);
```

**`Tensile/TensileCreateLibrary.py`**
```
archs = ['gfx'+''.join(map(str,arch)) for arch in globalParameters['SupportedISA'] \
if isSupported(arch)]
archFlags = ['--amdgpu-target=' + arch for arch in archs]    # hcc
archs = []
```
