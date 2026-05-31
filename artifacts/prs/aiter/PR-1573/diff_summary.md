# Diff summary

- **files changed:** 167 (diff was byte-capped; summary is partial)
- **lines:** +1559 / -1839
- **kernel-ish files:** 10

## Files (by churn)

- `csrc/include/mha_bwd.h`  (+205/-368)
- `csrc/cpp_itfs/mha_bwd.cpp`  (+533/-0)
- `csrc/py_itfs_cu/asm_mha_varlen_bwd.cu`  (+210/-255)
- `csrc/py_itfs_ck/mha_bwd_kernels.cu`  (+203/-237)
- `csrc/py_itfs_ck/mha_varlen_bwd_kernels.cu`  (+199/-240)
- `csrc/py_itfs_cu/asm_mha_bwd.cu`  (+179/-210)
- `hsa/gfx942/fmha_v3_bwd/codegen.py`  (+0/-327)
- `csrc/cpp_itfs/mha_bwd_generate.py`  (+0/-178)
- `aiter/jit/optCompilerConfig.json`  (+17/-15)
- `csrc/include/aiter_hip_common.h`  (+7/-5)
- `aiter/ops/mha.py`  (+6/-4)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a16_rtna.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a16_rtna_pddv.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a16_rtne.co`  (+0/-0)
- `hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a16_rtne_pddv.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/mha.py`**
```
from ..jit.core import AITER_CSRC_DIR, CK_DIR, AITER_META_DIR, compile_ops
f"{AITER_META_DIR}/hsa/codegen.py -m fmha_v3_bwd --output_dir {{}}",
"flags_extra_cc": ["'-DONLY_FAV3=0'"],
f"{AITER_META_DIR}/hsa/codegen.py -m fmha_v3_bwd --output_dir {{}}",
```

**`csrc/cpp_itfs/mha_bwd.cpp`**
```
namespace aiter {
std::tuple<int, int> get_padded_hdim(int hdim_q, int hdim_v, std::string arch_id)
if(hdim_q == 192 && hdim_v == 128 && arch_id == "gfx950")
return std::make_tuple(hdim_q, hdim_v);
```

**`csrc/include/aiter_hip_common.h`**
```
HIP_CALL(hipGetDeviceCount(&device_count));
if(device_count == 0)
hipDevice_t dev;
hipDeviceProp_t dev_prop;
```

**`csrc/include/mha_bwd.h`**
```
struct mha_bwd_args {
int mask_type; // 0: no mask   1: top_left_causal   2: bottom_right_causal   3: sliding_window
bool use_asm_v3;
bool v3_atomic_fp32;
```

**`csrc/py_itfs_ck/mha_bwd_kernels.cu`**
```
auto get_mask_type = [&]() {
if (mask.type == mask_enum::no_mask) {
return 0;
if (mask.type == mask_enum::window_generic) {
```
