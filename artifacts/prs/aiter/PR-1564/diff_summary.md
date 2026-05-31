# Diff summary

- **files changed:** 8
- **lines:** +18 / -16
- **kernel-ish files:** 8

## Files (by churn)

- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4.cuh`  (+9/-8)
- `aiter/ops/triton/mha.py`  (+2/-2)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`  (+2/-2)
- `aiter/ops/triton/pa_decode.py`  (+1/-1)
- `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`  (+1/-1)
- `csrc/cpp_itfs/pa/pa_ragged.cuh`  (+1/-1)
- `csrc/pybind/fused_mrope_rms_pybind.cu`  (+1/-1)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`  (+1/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/mha.py`**
```
rotary_cos/rotary_sin: Optional rotary embeddings (applied if provided) - interleaving flag unused here.
alibi_slopes: (nheads,) or (batch,nheads) bias slopes (currently ignored if provided - placeholder).
```

**`aiter/ops/triton/pa_decode.py`**
```
V1 for short sequences (<=8192), V2 with sequence partitioning for longer sequences.
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`**
```
int sorted_size = std::min(int64_t(tokens * topk * block_m.value()), sorted_token_ids.size(0));
int sorted_size = std::min(int64_t(tokens * topk * block_m.value()), sorted_token_ids.size(0));
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`**
```
4: kernelInstanceGEMM1(        64,       32,           32,       128,     1,       1,        3,),
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_mxfp4.cuh`**
```
static constexpr ck::index_t CShuffleNLane =
BLOCKSIZE == 64 ? NPerBlock / NXDLPerWave : NPerBlock / 2 / NXDLPerWave; // 64
<     Row,  Col,  DsLayout, ELayout,
AElementOp,  BElementOp, CDEElementOp,       GemmSpec,
```
