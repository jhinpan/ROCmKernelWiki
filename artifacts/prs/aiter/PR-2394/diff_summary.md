# Diff summary

- **files changed:** 11
- **lines:** +1075 / -25
- **kernel-ish files:** 11

## Files (by churn)

- `csrc/include/opus/opus.hpp`  (+307/-14)
- `op_tests/opus/device/test_opus_device.py`  (+255/-5)
- `op_tests/opus/device/test_wmma_f8.cu`  (+208/-0)
- `op_tests/opus/device/test_wmma_f32.cu`  (+152/-0)
- `op_tests/opus/device/test_wmma_f16.cu`  (+137/-0)
- `op_tests/opus/device/test_async_load.cu`  (+5/-0)
- `op_tests/opus/device/test_load_store_if.cu`  (+5/-0)
- `op_tests/opus/device/setup.py`  (+3/-0)
- `op_tests/opus/device/test_mfma_f16.cu`  (+1/-2)
- `op_tests/opus/device/test_mfma_f32.cu`  (+1/-2)
- `op_tests/opus/device/test_mfma_f8.cu`  (+1/-2)

## Key added lines (kernel files)

**`csrc/include/opus/opus.hpp`**
```
template<index_t rm = OPUS_FP32_to_BF16_DEFAULT> // gfx950/gfx1250 has instruction conversion, leave 'rm' here for compa
template<typename S, index_t sel = 0, std::enable_if_t<std::is_same_v<S, fp32x2_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp32_to_fp4_packed_x2(const S& s, float scale = 1.0f, number<sel> = {}) {
fp32x8_t v{s[0], s[1], 0, 0, 0, 0, 0, 0};
```

**`op_tests/opus/device/setup.py`**
```
"test_wmma_f16.cu",
"test_wmma_f32.cu",
"test_wmma_f8.cu",
```

**`op_tests/opus/device/test_async_load.cu`**
```
opus::s_wait_loadcnt(opus::number<0>{});
opus::s_wait_asynccnt(opus::number<0>{});
```

**`op_tests/opus/device/test_load_store_if.cu`**
```
s_wait_loadcnt(number<0>{});
s_wait_asynccnt(number<0>{});
```

**`op_tests/opus/device/test_opus_device.py`**
```
def run_wmma(self, A, B, C, variant):
fn = getattr(self._lib, f"run_wmma_{variant}")
fn.restype = None
fn.argtypes = [_VP, _VP, _VP, _I, _I, _I]
```
