# Diff summary

- **files changed:** 20
- **lines:** +107 / -160
- **kernel-ish files:** 2

## Files (by churn)

- `Tensile/Tests/pre_checkin/hgemm_asm_nn.yaml`  (+7/-64)
- `Tensile/Tests/pre_checkin/hgemm_hpa_asm_nn.yaml`  (+6/-65)
- `Tensile/SolutionStructs.py`  (+11/-4)
- `Tensile/Tests/pre_checkin/hgemm_asm_tn.yaml`  (+7/-3)
- `Tensile/Tests/pre_checkin/hgemm_hpa_iu2_asm_tt.yaml`  (+6/-3)
- `Tensile/Tests/nightly/assertions/test_hgemm_asem2_asm.yaml`  (+8/-0)
- `Tensile/Tests/pre_checkin/hgemm_asm_nt.yaml`  (+6/-2)
- `Tensile/Tests/pre_checkin/hgemm_asm_tt.yaml`  (+6/-2)
- `Tensile/Tests/pre_checkin/hgemm_hpa_asm_nt.yaml`  (+6/-2)
- `Tensile/Tests/pre_checkin/hgemm_hpa_asm_tn.yaml`  (+6/-2)
- `Tensile/Tests/pre_checkin/hgemm_hpa_asm_tt.yaml`  (+6/-2)
- `Tensile/Tests/pre_checkin/hgemm_hpa_iu2_asm_nn.yaml`  (+6/-2)
- `Tensile/Tests/pre_checkin/hgemm_hpa_iu2_asm_nt.yaml`  (+6/-2)
- `Tensile/Tests/pre_checkin/hgemm_hpa_iu2_asm_tn.yaml`  (+6/-2)
- `Tensile/Common.py`  (+4/-2)

## Key added lines (kernel files)

**`Tensile/Common.py`**
```
globalParameters["ArchCaps"] = {}
globalParameters["ArchCaps"][v] = {}
globalParameters["ArchCaps"][v]["HasEccHalf"] = (v==(9,0,6))
print1 ("# Arch caps for %s:%s" % (isaVersion, globalParameters["ArchCaps"][v]))
```

**`Tensile/SolutionStructs.py`**
```
and state["ProblemType"]["DataType"].isHalf():
if state["VectorWidth"] < 2:
reject(state, "VectorWidth must be >= 2 for half")
if globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"] and \
```
