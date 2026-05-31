# Diff summary

- **files changed:** 12
- **lines:** +59 / -149
- **kernel-ish files:** 8

## Files (by churn)

- `tile_engine/ops/gemm/validation_utils.py`  (+11/-39)
- `tile_engine/ops/gemm_preshuffle/commons/validation_utils.py`  (+9/-35)
- `tile_engine/ops/gemm/codegen_utils.py`  (+0/-32)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_codegen_utils.py`  (+0/-32)
- `tile_engine/ops/gemm/gemm_instance_builder.py`  (+9/-2)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_instance_builder.py`  (+9/-2)
- `tile_engine/ops/gemm_multi_d/gemm_multi_d_instance_builder.py`  (+8/-2)
- `tile_engine/ops/gemm/CMakeLists.txt`  (+5/-2)
- `tile_engine/ops/gemm/test_validation.py`  (+1/-3)
- `tile_engine/ops/gemm_preshuffle/CMakeLists.txt`  (+3/-0)
- `test/ck_tile/gemm_tile_engine/CMakeLists.txt`  (+2/-0)
- `tile_engine/ops/gemm_multi_d/CMakeLists.txt`  (+2/-0)

## Key added lines (kernel files)

**`tile_engine/ops/gemm/gemm_instance_builder.py`**
```
def __init__(self, working_path, gpu_target, datatype, layout, config_json=None):
self.gpu_target = gpu_target
self.gpu_target,
parser.add_argument(
```

**`tile_engine/ops/gemm/test_validation.py`**
```
gpu_name = "gfx90a"
```

**`tile_engine/ops/gemm/validation_utils.py`**
```
gpu_name: str,
gpu_name: str,
gpu_target: str,
if not validate_warp_configuration(warp_m, warp_n, warp_k, gpu_target):
```

**`tile_engine/ops/gemm_multi_d/gemm_multi_d_instance_builder.py`**
```
self.gpu_target = args.gpu_target
gpu_name = self.gpu_target
parser.add_argument(
"--gpu_target",
```

**`tile_engine/ops/gemm_preshuffle/commons/validation_utils.py`**
```
gpu_name: str,
gpu_target: str,
warp_tile_m,
warp_tile_n,
```
