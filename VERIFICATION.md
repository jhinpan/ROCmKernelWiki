# Hardware Verification — AMD Instinct gfx950 / CDNA4

> **Verified on:** 2026-06-01 · **GPU:** AMD Instinct MI350X, `gfx950:sramecc+:xnack-`
> · **Stack:** ROCm 7.2.0 / HIP 7.2.26015 / AMD clang 22.0.0git (`roc-7.2.0`)
> · **Method:** every claim below was checked by **compiling, running, and/or
> disassembling real code on the GPU** — not by reading docs. Each finding was
> then independently **re-run by a second adversarial pass** that tried to refute
> it (all agreed). Raw evidence excerpts are kept inline.

The active wiki is now scoped to gfx950 and gfx942. gfx950 claims combine
primary-source grounding with MI350X/MI355X execution; gfx942 remains
source-backed and compiler-checked until MI300 hardware is available. Toolchain
versions are separate evidence cells: a newer compiler confirms current
behavior but does not supersede version-sensitive results from an older stack.

## Reproducible MI355X harness — 2026-07-22

`validation/run.py` pinned runtime execution to MI355X device 0 under ROCm 7.2 /
clang 22 and retained a machine-readable bundle at
`validation/results/gfx950-mi355x-rocm720/`. It produced **26 pass, 0 fail, and 1
source-recorded** verdict. The suite covers device properties, direct-to-LDS
accepted widths and a 16-byte uniform-destination runtime check, permlane and
f8f6f4 availability, XF32 rejection, and HSA metadata extraction. gfx942 cells
are compile-only. The harness deliberately makes no cache, partition,
LDS-phase, numeric-MXFP, or performance claim.

## Additional pass — MI355X, 2026-07-20

This guide-sync pass repeated the relevant checks on an **AMD Instinct MI355X**
(`gfx950:sramecc+:xnack-`) with **ROCm 7.1.1, HIP 7.1.52802, and AMD clang 20
(`roc-7.1.1`)**. Tests were pinned to device 0 with
`ROCR_VISIBLE_DEVICES=0`; GPU utilization was checked before the empirical LDS
run. An initial run on a fully occupied node was rejected rather than used as
evidence, and the harness was rebuilt and rerun on an idle device.

### Device properties

| Property | MI355X measurement |
|---|---:|
| Compute Units | 256 |
| Wavefront width | 64 |
| Max threads/CU | 2048 (= 32 wave64 waves = 4 SIMD × 8 slots) |
| Max threads/block | 1024 (= 16 wave64 waves) |
| LDS/shared memory per CU | 163840 B (160 KiB) |
| Reported L2 | 4194304 B (4 MiB) |
| Global memory | 309220868096 B (288 GiB) |
| Memory bus width | 8192 bits |
| Reported memory clock | 2000000 kHz |
| Reported core clock | 2400000 kHz |

A separate kernel requested **102400 bytes (100 KiB)** of dynamic LDS, touched
both ends of the region, synchronized the workgroup, and returned the expected
value (`42`). This independently confirms usable capacity above gfx942's
64-KiB/CU limit, not merely a device-property string.

### Empirical LDS harness

The harness at
[`nod-ai/amd-shark-ai/docs/empirical-lds/harness`](https://github.com/nod-ai/amd-shark-ai/tree/efa471aeef66a260c85983cc41e833bfa769dade/docs/empirical-lds/harness)
was compiled with `LDS_GPU_ARCH=gfx950` and run without PMC counters, using its
`clock64()` classification signal.

| Check | Repeats | Result | Verdict |
|---|---:|---|---|
| `ds_read_b64` bank-count sweep (`1,2,4,8,16,32,64,96,128`) | 3 | high-latency bucket begins at 64 and recurs at 128; `most_likely_bank_count=64` | reproduced |
| `ds_read_b32`, 64 banks | 3 | phase `0–63` | reproduced |
| `ds_read_b64`, 64 banks | 5 | phases `0–31`, `32–63` | reproduced |
| `ds_read_b128`, 64 banks | 3 | noisy groups that did not match the upstream four-group table | **inconclusive**; table remains upstream-empirical |

The successful b64 sweep's average active-thread latency rose from 5224 cycles
at guess 32 to 10067 cycles at guess 64, fell to 5280 at 96, and rose to 10007
at 128. That periodic saturation is the harness's 64-bank signal. The b128
classifier result is intentionally not promoted to a verified fact.

### Architecture-specific transpose padding

The `block(32,32)` fp32 transpose example now selects one padding dword on
gfx942 and two on gfx950. This follows from the reproduced b32 phase model: a
gfx950 wave64 phase contains two adjacent columns across the same 32 rows, so
stride 34 separates them onto even/odd banks, whereas stride 33 overlaps the
sets. On MI355X the gfx950 example compiled with a 4352-byte static
group-segment payload, ran a 1024×2048 transpose, and passed exact comparison
(`max abs error: 0`). The same source device-compiled for gfx942 with the
expected 4224-byte payload. These are requested payload sizes; hardware
resource accounting rounds them to each target's LDS allocation unit. This
confirms target selection and functional correctness; the
zero-conflict conclusion is phase-table algebra rather than a fresh PMC count.

The architecture-specific b128 XOR example was also compiled and executed on
gfx950: for all four vector columns, its host checker enumerated every reported
16-lane phase and found 16 unique four-bank start slots. The guarded source
device-compiled for gfx942, whose eight-lane formula had already been enumerated
against all eight reported groups. This validates the example and its algebra,
not the still-inconclusive on-silicon b128 phase classifier.

### Raw-buffer OOB probe

A 64-lane kernel constructed a raw gfx9 buffer descriptor with
`NumRecords = 32` bytes and issued `raw_buffer_load_b32` at each lane's dword
offset. Lanes 0–7 returned the eight input values and lanes 8–63 returned zero;
all 64 checks passed on MI355X. A follow-up `raw_buffer_load_b128` beginning at
byte 24 of that 32-byte window returned the two in-range dwords followed by two
zeros, confirming per-component OOB handling for a partially crossing vector.
The same source cross-compiled for gfx942. The device ISA contains
`buffer_load_dword ... offen`.

This compile/run also corrected two pre-existing wiki-snippet problems:
ROCm 7.1.1 exposes the b32 builtin (not a `_f32` spelling), and a gfx9 raw-dword
descriptor needs initialized format flags (`0x00020000` in the probe). With
flags set to zero, every lane read as OOB. The run verifies predication on a
small window; the approximately 4-GiB maximum raw `NumRecords` extent remains an
ISA field-width fact rather than a >4-GiB allocation experiment.

The final target-selected helper was device-compiled for gfx942, gfx950, and
gfx1201. It uses the CK Tile word-3 split (`0x00020000` for these gfx9 targets,
`0x31004000` for gfx1201); only the gfx950 binary was executed in this pass.

### Non-temporal vector lowering

A native Clang `float __attribute__((ext_vector_type(4)))` load/store through
`__builtin_nontemporal_load/store` was device-compiled for gfx950 and gfx942.
Both ISA outputs contain `global_load_dwordx4 ... nt` and
`global_store_dwordx4 ... nt`. ROCm 7.1.1 rejected HIP's `float4` wrapper as the
builtin pointer type, so the wiki snippet now uses the compiler-native vector.
This verifies the clang-20 lowering, not a cross-version promise about which
cache level the `nt` policy affects.

### Cross-lane reduction and swap-result semantics

A one-wave kernel initialized lane `i` with `float(i + 1)` and ran both the
documented DPP-row + four-`ds_bpermute` reduction and the portable six-step
`__shfl_xor` baseline. Every lane returned **2080** from both paths.

The same MI355X run captured both results of the gfx950 swap builtins. With the
same lane-number value supplied as both operands, lane 0 observed
`permlane16_swap = {0,16}` and lane 16 observed `{0,16}`; likewise, lane 0
observed `permlane32_swap = {0,32}` and lane 32 observed `{0,32}`. Across all 64
lanes, selecting result element 1 in the lower half and element 0 in the upper
half produced exactly `lane ^ 16` / `lane ^ 32`. This caught the tempting but
incorrect assumption that result element 1 is always the partner. The guarded
source also device-compiled for gfx942 and emitted the expected
`ds_bpermute_b32` fallback path.

A review follow-up checked `ds_bpermute` address and `EXEC` behavior directly.
With every lane holding `lane + 1`, byte addresses `64 << 2` and `65 << 2`
returned 1 and 2 on every destination lane: source selection wraps modulo 64
rather than treating those logical indices as OOB. In a divergent branch where
source lane 1 was disabled in `EXEC`, selecting lane 1 returned 0. All checks
passed on MI355X; the same probe device-compiled for gfx942. This corrected the
earlier, unsafe “out-of-range source returns zero” wording.

### MI300X/gfx942 access limit

The configured MI300X SSH candidates all timed out on TCP/22. The VPN interface
was connected, but the required internal route/ACL was not available; a second
reachable host rejected all configured non-interactive credentials, so its GPU
could not be identified. A follow-up `ProxyJump` attempt through the reachable
MI355X node also timed out during the target SSH banner exchange. No host-key
check was bypassed and no MI300 remote command ran. Consequently this pass makes
**no MI300 runtime-verification
claim**: gfx942 phase tables remain upstream-empirical/official-source-backed
until authorized access is restored.

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
| 2 | `wiki/hardware/wavefront.md`, `wiki/techniques/{occupancy-tuning,vgpr-budgeting}.md`, `wiki/patterns/{low-occupancy,vgpr-pressure}.md`, `sources/docs/doc-rocm-hip-hw.md` | per-CU ceiling **40 waves (4×10)**; occupancy `min(10,…)` | Devices report **32 waves/CU (4×8)** (`Max Waves Per CU: 32` or `2048 threads / wave64`); 40/4×10 is a generic/older-GCN figure. All linked formulas and examples now use `min(8,…)` and max 32 waves/CU. |
| 3 | `wiki/migration/gfx942-to-gfx950.md` §4 (and table) | gfx942 direct-to-LDS **≤8 B**, "pipeline built on `dwordx2`" | gfx942 is **≤4 B (one dword)**; there is **no 8 B/`dwordx2`** form on either generation. (This also resolves a contradiction with `wiki/hardware/async-copy-lds.md`, which already said 4 B.) |
| 4 | `wiki/hardware/chiplet-xcd.md` | compute modes **SPX/CPX**; memory **NPS1/NPS4** | Four compute modes advertised: **SPX/DPX/QPX/CPX**. Memory caps on this MI350X: **NPS1/NPS2** — **NPS4 is not advertised** (it is an MI300-series layout). |
| 5 | `wiki/migration/gfx942-to-gfx950.md` §1 | FP8 `#if`-snippet had `#else #error` | Hit `#error` on the host compile pass; reworked the guard so the snippet compiles (host + both device passes). |
| 6 | `wiki/hardware/mxfp.md` | "TF32 … emulated via BF16; FP64 halved" | Sharpened (native xf32 intrinsic *fails to select*); FP64-halved tagged datasheet-derived. Noted that Clang exposes only the *scaled* f8f6f4 builtin. |
| 7 | 12 files | `../technique/…`, wrong-depth `../sources/…`, `technique-vgpr-budgeting.md` | **19 broken in-body relative links fixed.** (The validator only checked frontmatter ids, so these slipped through; see §5.) |

All new/edited code snippets were re-compiled on gfx950 **and** gfx942 (the
permlane block also executes on the MI350X) — the wiki's "every snippet
compilable" gate holds.

## 4. Notes / honest limits

- **LDS bank count (64).** The original 2026-06-01 single-CU microbenchmark did
  not distinguish 32 from 64 banks. The purpose-built upstream harness used in
  the 2026-07-20 pass did distinguish them through a repeated b64 stride sweep
  and reproduced b32/b64 phases. Its automatic b128 classification was still
  unstable, so the exact b128 table remains upstream-empirical.
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
