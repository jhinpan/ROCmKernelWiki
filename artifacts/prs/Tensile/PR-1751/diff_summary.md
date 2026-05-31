# Diff summary

- **files changed:** 33
- **lines:** +371 / -206
- **kernel-ish files:** 3

## Files (by churn)

- `Tensile/Tests/disabled/direct_to_lds/dtl_dgemm.yaml`  (+80/-40)
- `Tensile/Tests/extended/direct_to_lds/dtl_sgemm.yaml`  (+32/-16)
- `Tensile/Tests/extended/direct_to_lds/dtl_tsgr_sgemm.yaml`  (+24/-12)
- `Tensile/Tests/extended/direct_to_vgpr/dtv_igemm.yaml`  (+21/-12)
- `Tensile/Configs/miopen/archives/bert/2020-11-06/configs/nn.yaml`  (+18/-9)
- `Tensile/Configs/miopen/archives/bert/2020-11-06/configs/nt.yaml`  (+18/-9)
- `Tensile/Configs/miopen/archives/bert/2020-11-06/configs/tn.yaml`  (+18/-9)
- `Tensile/Tests/extended/direct_to_lds/dtl_dgemm.yaml`  (+16/-8)
- `Tensile/Tests/extended/direct_to_lds/dtl_hgemm.yaml`  (+16/-8)
- `Tensile/Tests/extended/direct_to_lds/dtl_tsgr_hgemm.yaml`  (+16/-8)
- `Tensile/Tests/extended/direct_to_vgpr/dtv_hgemm.yaml`  (+14/-8)
- `Tensile/Configs/miopen/archives/bert/2020-05-18/configs/bert_sgemm_xdlops_tn.yaml`  (+12/-8)
- `Tensile/Configs/miopen/archives/bert/2020-05-18/configs/bert_sgemm_xdlops_nn.yaml`  (+10/-5)
- `Tensile/Tests/disabled/direct_to_lds/dtl_dgemm_lite.yaml`  (+10/-5)
- `Tensile/SolutionStructs.py`  (+7/-6)

## Key added lines (kernel files)

**`Tensile/KernelWriter.py`**
```
if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"] or kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) \
and kernel["EnableMatrixInstruction"]:
```

**`Tensile/SolutionStructs.py`**
```
tcOther = "B" if tc == "A" else "A"
if state["PrefetchGlobalRead"] == 2 and state["DirectToLds%c"%tcOther] == False and state["DirectToVgpr%c"%tcOther] == F
reject(state, "DirectToLds%c does not work with PrefetchGlobalRead=2 and DirectToLds%c and DirectToVgpr%c "%(tc, tcOther
return False
```
