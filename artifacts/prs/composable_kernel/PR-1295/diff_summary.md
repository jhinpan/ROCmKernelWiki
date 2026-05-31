# Diff summary

- **files changed:** 16
- **lines:** +537 / -264
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck_tile/host/kernel_launch.hpp`  (+71/-131)
- `test/position_embedding/position_embedding.cpp`  (+62/-62)
- `example/ck_tile/01_fmha/utils.hpp`  (+96/-6)
- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+57/-22)
- `include/ck_tile/host/timer.hpp`  (+79/-0)
- `include/ck_tile/host/device_memory.hpp`  (+41/-18)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_tile_partitioner.hpp`  (+55/-4)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+29/-10)
- `example/ck_tile/01_fmha/generate.py`  (+18/-6)
- `include/ck_tile/host/stream_config.hpp`  (+17/-0)
- `include/ck_tile/ops/fmha/block/block_position_encoding.hpp`  (+3/-3)
- `include/ck_tile/core/config.hpp`  (+4/-0)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+2/-2)
- `example/ck_tile/01_fmha/README.md`  (+1/-0)
- `example/ck_tile/01_fmha/script/smoke_test.sh`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
"seqlen_q. if group-mode, means the average value of seqlen_q\n"
"total_seqlen_q = seqlen_q * batch, and seqlen_q per batch may vary\n"
"also with \"-s=s0,s1,s2...\" comma seperated int to set per batch seqlen(group-mode)")
.insert("s_kpad",
```

**`example/ck_tile/01_fmha/generate.py`**
```
TILE_PARTITIONER_MAP = {
"shb" : "ck_tile::FmhaFwdTilePartitioner_SHB",
"hbs" : "ck_tile::FmhaFwdTilePartitioner_HBS",
{F_squant},
```

**`example/ck_tile/01_fmha/utils.hpp`**
```
int32_t seqlen_avg,
int32_t seqlen_max = -1, // if not negative, clamp max
std::vector<int32_t> seqlens(
count, seqlen_max > 0 ? (seqlen_avg < seqlen_max ? seqlen_avg : seqlen_max) : seqlen_avg);
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
namespace impl {
template<index_t N, typename T> struct buffer_load_trait;
template<typename T> struct buffer_load_trait<16, T> { using payload_t = fp32x4_t; };
template<typename T> struct buffer_load_trait<8 , T> { using payload_t = fp32x2_t; };
```

**`include/ck_tile/host/device_memory.hpp`**
```
if(mMemSize != 0)
HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
mpDeviceBuf = nullptr;
if(mMemSize != 0)
```
