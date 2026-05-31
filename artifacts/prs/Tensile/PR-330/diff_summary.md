# Diff summary

- **files changed:** 32
- **lines:** +990 / -428
- **kernel-ish files:** 6

## Files (by churn)

- `Tensile/Tests/pre_checkin/test_sgemm_asm.yaml`  (+0/-270)
- `Tensile/KernelWriterAssembly.py`  (+74/-16)
- `Tensile/Tests/nightly/nonbatched/sgemm_asm_nn.yaml`  (+86/-0)
- `Tensile/Tests/pre_checkin/sgemm_asm_nn.yaml`  (+86/-0)
- `Tensile/Tests/nightly/nonbatched/sgemm_asm_nt.yaml`  (+85/-0)
- `Tensile/Tests/pre_checkin/sgemm_asm_nt.yaml`  (+85/-0)
- `Tensile/Tests/nightly/nonbatched/sgemm_asm_tn.yaml`  (+84/-0)
- `Tensile/Tests/pre_checkin/sgemm_asm_tn.yaml`  (+84/-0)
- `Tensile/Tests/nightly/nonbatched/sgemm_asm_tt.yaml`  (+80/-0)
- `Tensile/Tests/pre_checkin/sgemm_asm_tt.yaml`  (+80/-0)
- `Tensile/Tests/pre_checkin/hgemm_asm_nn.yaml`  (+0/-63)
- `Tensile/Tests/pre_checkin/hgemm_hpa_asm_nn.yaml`  (+1/-60)
- `Tensile/Tests/pre_checkin/test_pre_checkin.py`  (+38/-16)
- `Tensile/Tests/nightly/vector_width/hgemm_nn_asm.yaml`  (+44/-0)
- `Tensile/Tests/nightly/vector_width/hgemm_nn_source.yaml`  (+44/-0)

## Key added lines (kernel files)

**`Tensile/KernelWriterAssembly.py`**
```
if tensorChar == "C" and kernel["BufferStore"]:
elif indices[i] < kernel["ProblemType"]["NumIndicesC"] and not justOffset32:
elif indices[i] < kernel["ProblemType"]["NumIndicesC"] and not justOffset32:
"other stride mul d%u lower"%i)
```

**`Tensile/Source/Client.h`**
```
const unsigned int db = 0; // 0x1=header, 0x2=offset/value on each store, 0x4=loop debug
```

**`Tensile/Tests/create_tests.py`**
```
for f in glob.glob("%s/*aml"%targetDir):
```

**`Tensile/Tests/nightly/nonbatched/test_nonbatched.py`**
```
import Tensile.Tensile as Tensile
def test_sgemm_asm_nt(tmpdir):
Tensile.Tensile([Tensile.TensileTestPath("nightly/nonbatched/sgemm_asm_nt.yaml"), tmpdir.strpath])
def test_sgemm_asm_nn(tmpdir):
```

**`Tensile/Tests/nightly/vector_width/test_vector_width.py`**
```
import Tensile.Tensile as Tensile
def test_sgemm_nn_source(tmpdir):
Tensile.Tensile([Tensile.TensileTestPath("nightly/vector_width/sgemm_nn_source.yaml"), tmpdir.strpath])
def test_sgemm_nn_asm(tmpdir):
```
