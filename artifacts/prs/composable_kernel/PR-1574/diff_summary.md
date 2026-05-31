# Diff summary

- **files changed:** 32
- **lines:** +879 / -516
- **kernel-ish files:** 31

## Files (by churn)

- `include/ck/utility/type.hpp`  (+313/-316)
- `codegen/test/rtc/src/compile_kernel.cpp`  (+199/-4)
- `codegen/test/gemm_multiple_d.cpp`  (+13/-119)
- `include/ck/utility/data_type.hpp`  (+122/-6)
- `codegen/test/batched_gemm_softmax_gemm.cpp`  (+87/-0)
- `codegen/test/include/common.hpp`  (+33/-16)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp`  (+19/-8)
- `include/ck/utility/math_v2.hpp`  (+19/-1)
- `include/ck/utility/enable_if.hpp`  (+8/-10)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_xdl_cshuffle.hpp`  (+8/-2)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp`  (+7/-1)
- `include/ck/utility/loop_scheduler.hpp`  (+4/-4)
- `include/ck/tensor_operation/gpu/device/device_base.hpp`  (+4/-3)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp`  (+4/-3)
- `codegen/test/rtc/CMakeLists.txt`  (+6/-0)

## Key added lines (kernel files)

**`codegen/test/batched_gemm_softmax_gemm.cpp`**
```
using half = _Float16;
const std::string gemm_compile_check = R"__ck__(
extern "C" __global__ void f(const ck::half_t* a, const ck::half_t* b, const ck::half_t* b1, ck::half_t* c) {
using G = ${template};
```

**`codegen/test/gemm_multiple_d.cpp`**
```
constexpr auto desc = G::make_descriptor(ck::make_naive_tensor_descriptor_packed(ck::make_tuple(${m}, ${k})),
auto solutions = prob.GetSolutions("gfx90a", prologue, epilogue);
std::cout << "Num solutions: " << solutions.size() << std::endl;
for(auto i = 0; i < solutions.size(); ++i)
```

**`codegen/test/include/common.hpp`**
```
inline std::vector<rtc::src_file> create_headers_for_test()
auto ck_headers = ck::host::GetHeaders();
std::transform(ck_headers.begin(), ck_headers.end(), std::back_inserter(result), [](auto& p) {
std::string content;
```

**`codegen/test/rtc/include/rtc/compile_kernel.hpp`**
```
src_file(std::filesystem::path p, std::string c) : path{std::move(p)}, content{std::move(c)} {}
std::string content;
kernel compile_kernel(const std::vector<src_file>& srcs,
```

**`codegen/test/rtc/src/compile_kernel.cpp`**
```
bool EndsWith(const std::string& value, const std::string& suffix)
if(suffix.size() > value.size())
return false;
return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
```
