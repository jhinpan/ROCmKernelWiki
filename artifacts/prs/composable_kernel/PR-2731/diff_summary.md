# Diff summary

- **files changed:** 17
- **lines:** +3272 / -2
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp`  (+1198/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp`  (+603/-0)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp`  (+519/-0)
- `example/ck_tile/01_fmha/example_fmha_fwd_v3.cpp`  (+492/-0)
- `example/ck_tile/01_fmha/fmha_fwd_v3_impl.hpp`  (+159/-0)
- `example/ck_tile/01_fmha/fmha_fwd_v3.hpp`  (+67/-0)
- `example/ck_tile/01_fmha/fmha_fwd_v3.cpp`  (+60/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp`  (+44/-0)
- `example/ck_tile/01_fmha/script/benchmark_fwd_v3.sh`  (+31/-0)
- `example/ck_tile/01_fmha/CMakeLists.txt`  (+22/-0)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp`  (+16/-0)
- `example/ck_tile/01_fmha/instances/fmha_fwd_v3_d128_bf16_mask.cpp`  (+14/-0)
- `example/ck_tile/01_fmha/instances/fmha_fwd_v3_d128_bf16_nmask.cpp`  (+14/-0)
- `example/ck_tile/01_fmha/instances/fmha_fwd_v3_d128_fp16_mask.cpp`  (+14/-0)
- `example/ck_tile/01_fmha/instances/fmha_fwd_v3_d128_fp16_nmask.cpp`  (+14/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/example_fmha_fwd_v3.cpp`**
```
auto parse_cmd_args(int argc, char* argv[]) -> std::pair<bool, ck_tile::ArgParser>
ck_tile::ArgParser arg_parser;
arg_parser.insert("prec", "fp16", "data type. fp16/bf16")
.insert("b", "2", "batch size")
```

**`example/ck_tile/01_fmha/fmha_fwd_v3.cpp`**
```
namespace ck_tile {
std::ostream& operator<<(std::ostream& stream, const fmha_fwd_v3_args::data_type_enum& data_type)
switch(data_type)
case fmha_fwd_v3_args::data_type_enum::fp16: return stream << "fp16";
```

**`example/ck_tile/01_fmha/fmha_fwd_v3.hpp`**
```
namespace ck_tile {
struct fmha_fwd_v3_args
enum class data_type_enum
data_type_enum data_type;
```

**`example/ck_tile/01_fmha/fmha_fwd_v3_impl.hpp`**
```
template <>                                                                                \
std::pair<bool, float> fmha_fwd_v3_kernel_dispatch<kernel_traits>(                         \
const fmha_fwd_v3_args& args, const stream_config& config)                             \
{                                                                                          \
```

**`example/ck_tile/01_fmha/instances/fmha_fwd_v3_d128_bf16_mask.cpp`**
```
namespace ck_tile {
using kernel_traits =
fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::bf16, false, true>;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits)
```
