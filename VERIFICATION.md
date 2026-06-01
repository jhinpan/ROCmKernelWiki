# Hardware Verification — AMD Instinct MI350X (gfx950 / CDNA4)

> **Verified on:** 2026-06-01 · **GPU:** AMD Instinct MI350X, `gfx950:sramecc+:xnack-`
> · **Stack:** ROCm 7.2.0 / HIP 7.2.26015 / AMD clang 22.0.0git (`roc-7.2.0`)
> · **Method:** every claim below was checked by **compiling, running, and/or
> disassembling real code on the GPU** — not by reading docs. Each finding was
> then independently **re-run by a second adversarial pass** that tried to refute
> it (all agreed). Raw evidence excerpts are kept inline.

The wiki content was originally authored against **ROCm 7.0.2** on an **RDNA4
(gfx1201)** box, where the CDNA-MFMA paths could only be *cross-compiled* and the
hardware facts traced to ISA docs/whitepapers. This pass re-grounds the
gfx950-specific claims and the runnable examples against **actual MI350X silicon**
on a newer stack (ROCm 7.2 > the pinned 7.0.2, so confirmations are strictly
stronger). It corrects the handful of claims the silicon contradicted.

## 1. Device specs — confirmed on silicon

`rocminfo` + `hipGetDeviceProperties` on this MI350X:

| Property | Wiki | Measured | |
|---|---|---|---|
| Compute Units | 256 | `multiProcessorCount = 256` | ✅ |
| Shader Engines | 32 | `Shader Engines: 32` | ✅ |
| SIMDs / CU | 4 | `SIMDs per CU: 4` | ✅ |
| Wavefront | wave64 | `warpSize = 64` | ✅ |
| L1 / CU | 32 KB | `L1: 32 KB` | ✅ |
| L2 / XCD | 4 MB | `l2CacheSize = 4096 KB` | ✅ |
| Infinity Cache (MALL) | 256 MB | `L3: 262144 KB` | ✅ |
| HBM3E | 288 GB | `totalGlobalMem = 309.22 GB` (= 288 GiB) | ✅ |
| Max clock | 2200 MHz | `clockRate = 2200 MHz` | ✅ |
| **LDS / CU** | **160 KB** | `sharedMemPerBlock = 160 KB`; a ~100 KB dynamic `extern __shared__` kernel launches & runs (would fail on gfx942's 64 KB) | ✅ |
| XCD topology | 8 XCDs × 32 active CU | `amd-smi`: SPX=8 XCC, CPX=8×1 XCC ⇒ 256/8 = 32 | ✅ |

## 2. gfx950 ISA / numeric claims — confirmed on silicon

| Claim | Evidence | |
|---|---|---|
| LDS **160 KB / 64 banks** vs gfx942 64 KB / 32 banks | capacity confirmed via props + dynamic-LDS launch; bank count per CDNA4 ISA (a `clock64()` microbench did *not* surface conflict serialization — see §4 note) | ✅ |
| **FP8 OCP (gfx950) ≠ FNUZ (gfx942)**, not bit-compatible | on-device: OCP-E4M3 max finite = **448**, FNUZ-E4M3 max = **240**; the same float encodes to different bytes and the same byte decodes to different reals under the two interpretations | ✅ |
| **f8f6f4 unified MFMA + MX (E8M0) block scaling** | `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` / `_32x32x64_f8f6f4` (A/B = `i32x8`, acc = `f32x4`, 9 args) compile & emit `v_mfma_scale_f32_*_f8f6f4` on gfx950; absent on gfx942; a scaled-GEMM kernel runs without fault | ✅ |
| **TF32/XF32 matrix path dropped on gfx950** | `__builtin_amdgcn_mfma_f32_16x16x8_xf32` (and `32x32x4`) compiles for gfx942 but on gfx950 hard-fails: `fatal error: Cannot select: intrinsic llvm.amdgcn.mfma.f32.16x16x8.xf32`. **Sharper than "emulated via BF16":** the native intrinsic does not lower at all. | ✅ |
| **Direct-to-LDS widens to 16 B on gfx950** | `__builtin_amdgcn_load_to_lds` legal byte sizes = `{1,2,4}` on gfx942 vs `{1,2,4,12,16}` on gfx950; flat form emits `global_load_lds_dword{,x3,x4}`; completion on VMCNT | ✅ |
| FP64 matrix opcode still present | `mfma_f64_16x16x4` compiles on both gfx942 and gfx950 (the per-CU *throughput* halving is a datasheet figure, not measured here) | ⚠️ source-reported |

## 3. Corrections applied (silicon contradicted the wiki)

| # | File | Was | Now (verified) |
|---|---|---|---|
| 1 | `wiki/hardware/cross-lane.md`, `wiki/migration/gfx942-to-gfx950.md` §5 | gfx950 cross-lane uses `__builtin_amdgcn_permlanex16(v,v,sel0,sel1,…)` | That RDNA selector-form needs `gfx10-insts` and **does not compile for CDNA4**. The real gfx950 op is `v_permlane16_swap_b32` via `__builtin_amdgcn_permlane16_swap` (+ `permlane32_swap`); takes only `fi`/`bound_ctrl`, no selectors. |
| 2 | `wiki/hardware/wavefront.md` | per-CU ceiling **40 waves (4×10)**; occupancy `min(10,…)` | Device reports **32 waves/CU (4×8)** (`Max Waves Per CU: 32`); 40/4×10 is a pre-CDNA (GCN/Vega) figure. Formula → `min(8,…)`, `max 32 waves/CU`. |
| 3 | `wiki/migration/gfx942-to-gfx950.md` §4 (and table) | gfx942 direct-to-LDS **≤8 B**, "pipeline built on `dwordx2`" | gfx942 is **≤4 B (one dword)**; there is **no 8 B/`dwordx2`** form on either generation. (This also resolves a contradiction with `wiki/hardware/async-copy-lds.md`, which already said 4 B.) |
| 4 | `wiki/hardware/chiplet-xcd.md` | compute modes **SPX/CPX**; memory **NPS1/NPS4** | Four compute modes advertised: **SPX/DPX/QPX/CPX**. Memory caps on this MI350X: **NPS1/NPS2** — **NPS4 is not advertised** (it is an MI300-series layout). |
| 5 | `wiki/migration/gfx942-to-gfx950.md` §1 | FP8 `#if`-snippet had `#else #error` | Hit `#error` on the host compile pass; reworked the guard so the snippet compiles (host + both device passes). |
| 6 | `wiki/hardware/mxfp.md` | "TF32 … emulated via BF16; FP64 halved" | Sharpened (native xf32 intrinsic *fails to select*); FP64-halved tagged datasheet-derived. Noted that Clang exposes only the *scaled* f8f6f4 builtin. |
| 7 | 12 files | `../technique/…`, wrong-depth `../sources/…`, `technique-vgpr-budgeting.md` | **19 broken in-body relative links fixed.** (The validator only checked frontmatter ids, so these slipped through; see §5.) |

All new/edited code snippets were re-compiled on gfx950 **and** gfx942 (the
permlane block also executes on the MI350X) — the wiki's "every snippet
compilable" gate holds.

## 4. Notes / honest limits

- **LDS bank count (64).** Capacity (160 KB) is hardware-confirmed. The 64-bank
  geometry is from the CDNA4 ISA; a single-CU `clock64()` strided-`ds_read`
  microbench could **not** distinguish 32- vs 64-bank conflicts (serialization is
  hidden behind issue/latency at that scale) — the reliable signal is the
  `SQ_LDS_BANK_CONFLICT` performance counter, not wall-clock. Treat the bank
  *timing* sensitivity claim as counter-observable, not stopwatch-observable.
- **FP64 throughput halving** and the **MI300X row** of the topology table are
  datasheet/whitepaper figures (no gfx942 hardware on this host to measure).

## 5. Examples — now RAN on MI350X (previously cross-compile-only for gfx950)

All 12 examples build with `--offload-arch=gfx950` **and execute** on this GPU.

| Example | builds | runs | self-check | measured on gfx950 |
|---|---|---|---|---|
| bandwidth-microbench | ✅ | ✅ | **PASS** | ~6.2–6.3 TB/s sustained HBM3E read |
| vector-add-asm | ✅ | ✅ | **PASS** | 6785 GB/s effective BW (HIP part); asm part assembles |
| transpose-lds | ✅ | ✅ | **PASS** (exact) | 4114 GB/s effective |
| rmsnorm | ✅ | ✅ | **PASS** (5 cases) | — |
| paged-attention | ✅ | ✅ | **PASS** | 0.045 ms |
| mla-decode | ✅ | ✅ | **PASS** | — |
| fused-moe | ✅ | ✅ | **PASS** | 122 µs/iter |
| flash-attention-ck | ✅ | ✅ | **PASS** | fp32 portable |
| ck-hgemm | ✅ | ✅ | **PASS** | rocWMMA FP16 256³, max abs err 0 |
| grouped-gemm | ✅ | ✅ | **PASS** | rocWMMA, warpSize=64 |
| flydsl-preshuffle-gemm | ✅ | ✅ | **PASS** | rocWMMA 256³ preshuffle, exact |
| **fp8-gemm** | ✅ | ✅ | **none** ⚠️ | `main()` never launches the kernel — it only compiles & verifies that `v_mfma_scale_f32_16x16x128_f8f6f4` is emitted; it does **not** run a numeric FP8 GEMM check. README/wiki should not imply a runtime correctness check for this one. |

## Reproduce

```bash
rocminfo | grep -E 'gfx950|Compute Unit|Wavefront|Cache|Max Waves'
hipcc --offload-arch=gfx950 -O2 <snippet>.cpp -o t && ./t        # run on gfx950
hipcc --offload-arch=gfx950 -O2 -S <snippet>.cpp -o t.s          # inspect emitted ISA
hipcc --offload-arch=gfx942 -O2 -c <snippet>.cpp -o t.o          # cross-compile check
```
