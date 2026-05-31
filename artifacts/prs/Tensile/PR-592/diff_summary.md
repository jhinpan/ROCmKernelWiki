# Diff summary

- **files changed:** 20
- **lines:** +1303 / -96
- **kernel-ish files:** 10

## Files (by churn)

- `Tensile/Configs/rocblas_cgemm_hip_lite.yaml`  (+364/-0)
- `Tensile/Source/TensileTypes.h`  (+118/-2)
- `Tensile/KernelWriterSource.py`  (+45/-48)
- `Tensile/Tests/pre_checkin/float_complex/cgemm_hip_source_cn.yaml`  (+73/-0)
- `Tensile/Tests/pre_checkin/float_complex/cgemm_hip_source_tn.yaml`  (+72/-0)
- `Tensile/Tests/pre_checkin/float_complex/cgemm_hip_source_nc.yaml`  (+71/-0)
- `Tensile/Tests/pre_checkin/float_complex/cgemm_hip_source_nt.yaml`  (+70/-0)
- `Tensile/Tests/pre_checkin/float_complex/cgemm_hip_source_nn.yaml`  (+69/-0)
- `Tensile/Tests/pre_checkin/float_complex/cgemm_hip_source_cc.yaml`  (+66/-0)
- `Tensile/Tests/pre_checkin/float_complex/cgemm_hip_source_ct.yaml`  (+65/-0)
- `Tensile/Tests/pre_checkin/float_complex/cgemm_hip_source_tc.yaml`  (+65/-0)
- `Tensile/Tests/pre_checkin/float_complex/cgemm_hip_source_tt.yaml`  (+64/-0)
- `Tensile/Source/Client.h`  (+49/-3)
- `Tensile/Source/ReferenceCPU.h`  (+23/-24)
- `Tensile/Source/MathTemplates.cpp`  (+44/-0)

## Key added lines (kernel files)

**`Tensile/DataType.py`**
```
{'char': 'C', 'name': 'complexSingle', 'enum': 'ComplexFloat', 'reg': 2, 'ocl': 'float2', 'hip': 'TensileComplexFloat', 
{'char': 'Z', 'name': 'complexDouble', 'enum': 'ComplexDouble', 'reg': 4, 'ocl': 'double2', 'hip': 'TensileComplexDouble
def isDoubleComplex(self):
return self.value == DataType.complexDouble
```

**`Tensile/KernelWriterSource.py`**
```
kStr += "template <typename T>%s" % (self.endLine)
if kernel["GlobalSplitU"] > 1: # 1st kernel will have taken care of B
if kernel["ProblemType"]["UseBeta"]:
kStr += "#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) atomicAddType(&(DST), (ALPHA)*(REG));"
```

**`Tensile/SolutionWriter.py`**
```
s += "%sbool betaZero = beta == (%s)0;\n" % (t, typeName)
```

**`Tensile/Source/Client.h`**
```
initialData[serialIdx] = val; // actually initialize the element
val += static_cast<DataType>(1);
template<typename DataType, typename std::enable_if<!(std::is_same<DataType, TensileComplexFloat>{} || std::is_same<Data
template<typename DataType, typename std::enable_if<std::is_same<DataType, TensileComplexFloat>{} || std::is_same<DataTy
```

**`Tensile/Source/MathTemplates.cpp`**
```
template< >
float tensileMultiply( TensileHalf a, TensileHalf b ) {
return (float)a * (float)b;
template< >
```
