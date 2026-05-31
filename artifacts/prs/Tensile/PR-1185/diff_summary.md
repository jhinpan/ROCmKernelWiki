# Diff summary

- **files changed:** 8 (diff was byte-capped; summary is partial)
- **lines:** +6020 / -12
- **kernel-ish files:** 3

## Files (by churn)

- `Tensile/Configs/mfma/mfma_igemm_nn_asm_full.yaml`  (+1682/-0)
- `Tensile/Configs/mfma/mfma_igemm_tn_asm_full.yaml`  (+1619/-0)
- `Tensile/Configs/mfma/mfma_igemm_nt_asm_full.yaml`  (+1405/-0)
- `Tensile/Configs/mfma/mfma_igemm_tt_asm_full.yaml`  (+1167/-0)
- `Tensile/Configs/mfma/mfma_igemm_lite_test.yaml`  (+94/-0)
- `Tensile/Common.py`  (+48/-11)
- `HostLibraryTests/DataTypes_test.cpp`  (+4/-0)
- `Tensile/Code.py`  (+1/-1)

## Key added lines (kernel files)

**`HostLibraryTests/DataTypes_test.cpp`**
```
std::tuple<int8_t>,
static_assert(Tensile::TypeInfo<int8_t>::Enum == Tensile::DataType::Int8, "Int8");
static_assert(Tensile::TypeInfo<int8_t>::Packing == 1, "Int8");
Tensile::DataType::Int8,
```

**`Tensile/Common.py`**
```
globalParameters["EnableAsserts"] = False         # Enable assembly debug assert
globalParameters["EnableDebugA"] = False          # Enable / Disable CheckValue1A
globalParameters["EnableDebugB"] = False          # Enable / Disable CheckValue1B
globalParameters["EnableDebugC"] = False          # Enable / Disable CheckValueC
```
