# Diff summary

- **files changed:** 27
- **lines:** +3726 / -11
- **kernel-ish files:** 23

## Files (by churn)

- `include/ck_tile/core/arch/arch.hpp`  (+754/-8)
- `test/ck_tile/core/arch/mma/test_amdgcn_mma.cpp`  (+682/-0)
- `test/ck_tile/core/arch/test_arch.cpp`  (+396/-0)
- `include/ck_tile/core/config.hpp`  (+239/-0)
- `include/ck_tile/core/arch/mma/mma.hpp`  (+234/-0)
- `include/ck_tile/core/arch/mma/mfma/mfma_selector.hpp`  (+189/-0)
- `include/ck_tile/core/arch/mma/mfma/mfma_gfx9.hpp`  (+162/-0)
- `include/ck_tile/core/arch/mma/wmma/wmma_selector.hpp`  (+161/-0)
- `include/ck_tile/core/arch/mma/mma_traits.hpp`  (+151/-0)
- `include/ck_tile/core/arch/mma/amdgcn_mma.hpp`  (+118/-0)
- `include/ck_tile/core/arch/mma/wmma/wmma_transforms.hpp`  (+112/-0)
- `include/ck_tile/core/arch/mma/wmma/wmma_gfx11.hpp`  (+109/-0)
- `include/ck_tile/core/arch/mma/wmma/wmma_gfx12.hpp`  (+69/-0)
- `include/ck_tile/core/arch/mma/mma_selector.hpp`  (+63/-0)
- `include/ck_tile/core/arch/mma/mma_transforms.hpp`  (+48/-0)

## Key added lines (kernel files)

**`include/ck_tile/core/arch/arch.hpp`**
```
namespace core::arch {
enum struct amdgcn_target_id
GFX908         = 0x0908, // MI-100...
GFX90A         = 0x090A,
```

**`include/ck_tile/core/arch/mma/amdgcn_mma.hpp`**
```
namespace ck_tile::core::arch::mma {
struct Unsupported;
template <typename MmaOp>
concept MmaOpI = requires(MmaOp op) {
```

**`include/ck_tile/core/arch/mma/mfma/mfma_gfx9.hpp`**
```
namespace ck_tile::core::arch::mma {
struct DefaultMfmaCtrlFlags
static constexpr uint32_t Cbsz = 0; // CBSZ flag, default 0
static constexpr uint32_t Abid = 0; // ABID flag, default 0
```

**`include/ck_tile/core/arch/mma/mfma/mfma_selector.hpp`**
```
namespace ck_tile::core::arch::mma {
implementation.
template <typename ADataType,
typename BDataType,
```

**`include/ck_tile/core/arch/mma/mfma/mfma_traits.hpp`**
```
namespace ck_tile::core::arch::mma {
struct MfmaOp;
template <typename MmaOp, typename = void>
struct is_mma_op_mfma : std::false_type
```
