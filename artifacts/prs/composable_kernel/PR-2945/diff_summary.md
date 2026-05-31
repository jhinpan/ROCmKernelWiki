# Diff summary

- **files changed:** 11
- **lines:** +397 / -620
- **kernel-ish files:** 10

## Files (by churn)

- `test/ck_tile/fmha/test_fmha_bwd.inc`  (+0/-347)
- `test/ck_tile/fmha/test_fmha_bwd.cpp`  (+248/-0)
- `test/ck_tile/fmha/test_fmha_fwd.cpp`  (+102/-10)
- `test/ck_tile/fmha/CMakeLists.txt`  (+47/-32)
- `test/ck_tile/fmha/test_fmha_fwd_bf16.cpp`  (+0/-44)
- `test/ck_tile/fmha/test_fmha_fwd_fp16.cpp`  (+0/-44)
- `test/ck_tile/fmha/test_fmha_fwd_fp8.cpp`  (+0/-42)
- `test/ck_tile/fmha/test_fmha_fwd_fp32.cpp`  (+0/-39)
- `test/ck_tile/fmha/test_fmha_bwd_bf16.cpp`  (+0/-21)
- `test/ck_tile/fmha/test_fmha_bwd_fp16.cpp`  (+0/-21)
- `test/ck_tile/fmha/test_fmha_bwd_fp32.cpp`  (+0/-20)

## Key added lines (kernel files)

**`test/ck_tile/fmha/test_fmha_bwd.cpp`**
```
using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
```

**`test/ck_tile/fmha/test_fmha_fwd.cpp`**
```
template <typename T>
struct TestConfigs
static constexpr auto HDimValues = std::array{
std::tuple{32, -1},
```
