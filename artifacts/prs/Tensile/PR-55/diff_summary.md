# Diff summary

- **files changed:** 14 (diff was byte-capped; summary is partial)
- **lines:** +6576 / -25
- **kernel-ish files:** 13

## Files (by churn)

- `clients/assembly/archive/sgemm_NT_128x128x4.s`  (+924/-0)
- `clients/assembly/archive/sgemm_NT_128x128x8_sideways.s`  (+921/-0)
- `clients/assembly/archive/sgemm_NT_128x128x8_branch.s`  (+789/-0)
- `clients/assembly/archive/sgemm_NT_128x128x8_c.s`  (+722/-0)
- `clients/assembly/archive/sgemm_NT_128x128x8.s`  (+706/-0)
- `clients/assembly/archive/sgemm_NT_128x128x8_d.s`  (+701/-0)
- `clients/assembly/archive/sgemm_NT_128x128x8_b.s`  (+686/-0)
- `clients/assembly/archive/sgemm_NT_128x128x8_75.s`  (+633/-0)
- `clients/assembly/archive/fmac.s`  (+292/-0)
- `clients/assembly/archive/sgemm_NT_128x128x8_working.s`  (+105/-0)
- `TensileGen/SolutionSelectionWriter.py`  (+73/-23)
- `clients/assembly/README.txt`  (+15/-0)
- `TensileBenchmark/TensileBenchmark.cpp`  (+8/-1)
- `TensileLib/include/Tensile.h`  (+1/-1)

## Key added lines (kernel files)

**`TensileBenchmark/TensileBenchmark.cpp`**
```
printf("Status: initControls()\n");
printf("Status: calling hipGetDeviceCount()\n");
printf("Status: calling hipSetDevice()\n");
printf("Status: calling tensileCreateEmptyControl()\n");
```

**`TensileGen/SolutionSelectionWriter.py`**
```
self.printDebugLib = True
if self.printDebugLib: s += "  printf(\"Tensile::" + functionName + "()\\n\");\n"
if self.printDebugLib: s += "  printf(\"%s\\n\", problem.toString().c_str() );\n"
if self.printDebugLib: s += "  printf(\"Tensile::" + functionName + "()\\n\");"
```

**`TensileLib/include/Tensile.h`**
```
enum { maxQueues = 4 } maxQueues_;
```

**`clients/assembly/archive/fmac.s`**
```
.hsa_code_object_version 2,0
.hsa_code_object_isa 8, 0, 3, "AMD", "AMDGPU"
.p2align 8
.amdgpu_hsa_kernel sgemm_NT
```

**`clients/assembly/archive/sgemm_NT_128x128x4.s`**
```
.macro ZERO_REGISTERS
.set c, 64
v_mov_b32 v[c+ 0], 0
v_mov_b32 v[c+ 1], 0
```
