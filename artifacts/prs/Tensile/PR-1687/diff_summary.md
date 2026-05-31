# Diff summary

- **files changed:** 6 (diff was byte-capped; summary is partial)
- **lines:** +772 / -828
- **kernel-ish files:** 6

## Files (by churn)

- `Tensile/CustomKernels/DGEMM_Aldebaran_NN_MT128x128x16_MI16x16x4x1_GRVW2_SU4_SUS128_WGM4.s`  (+742/-742)
- `Tensile/Components/MAC_F32.py`  (+5/-60)
- `Tensile/Common.py`  (+14/-15)
- `Tensile/Components/Signature.py`  (+5/-5)
- `HostLibraryTests/hip/HipSolutionAdapter_test.cpp`  (+4/-4)
- `HostLibraryTests/ocl/OclSolutionAdapter_test.cpp`  (+2/-2)

## Key added lines (kernel files)

**`HostLibraryTests/hip/HipSolutionAdapter_test.cpp`**
```
k.args.append<uint64_t>("offsetD", desc.offset());
k.args.append<uint64_t>("offsetC", desc.offset());
k.args.append<uint64_t>("offsetD", desc.offset());
k.args.append<uint64_t>("offsetC", desc.offset());
```

**`HostLibraryTests/ocl/OclSolutionAdapter_test.cpp`**
```
k.args.append<uint64_t>("offsetD", desc.offset());
k.args.append<uint64_t>("offsetC", desc.offset());
```

**`Tensile/Common.py`**
```
(8, 0, 3): {'SupportedISA': True, 'HasExplicitCO': False, 'HasExplicitNC': False, 'HasDirectToLdsDest': False, 'HasDirec
(9, 0, 0): {'SupportedISA': True, 'HasExplicitCO': True, 'HasExplicitNC': False, 'HasDirectToLdsDest': False, 'HasDirect
(9, 0, 6): {'SupportedISA': True, 'HasExplicitCO': True, 'HasExplicitNC': False, 'HasDirectToLdsDest': False, 'HasDirect
(9, 0, 8): {'SupportedISA': True, 'HasExplicitCO': True, 'HasExplicitNC': False, 'HasDirectToLdsDest': False, 'HasDirect
```

**`Tensile/Components/MAC_F32.py`**
```
if instruction == "v_fma_f32":
kStr += "v_fma_f32 {cStr}, {aStr}, {bStr}, {cStr}{endLine}".format_map(vars)
kStr += "{instruction} {cStr}, {aStr}, {bStr}{endLine}".format_map(vars)
kStr += priority(writer, 1, "Raise priority while processing macs")
```

**`Tensile/Components/Signature.py`**
```
kStr += self.addArgument("OffsetD", '8', offset, "by_value", "u64"); offset += 8
kStr += self.addArgument("OffsetC", '8', offset, "by_value", "u64"); offset += 8
kStr += self.addArgument("OffsetA", '8', offset, "by_value", "u64"); offset += 8
kStr += self.addArgument("OffsetB", '8', offset, "by_value", "u64"); offset += 8
```
