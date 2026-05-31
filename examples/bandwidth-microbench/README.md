# bandwidth-microbench — portable float4 HBM read bandwidth

Pure-HIP, **runnable** cousin of gcnasm's `bandwidth_memread`. A persistent
grid-stride kernel streams a large array with **128-bit (`float4`) non-temporal
loads** and accumulates into a register sink so the compiler cannot dead-code
the loads away. It sweeps several working-set sizes and reports measured GB/s.

## What it shows

- 128-bit vectorized loads: the ISA emits `global_load_b128 ... th:TH_LOAD_NT`
  (verified by `build.sh`), i.e. wide + non-temporal streaming reads.
- Persistent grid (a couple of blocks per CU) + manual `UNROLL=8` so several
  loads are in flight before the first `vmcnt` wait.
- A self-check kernel reduces a known buffer (every scalar = 1.0f) and compares
  the sum against a CPU reference, proving the load path is correct.

## Classification

**PORTABLE** — pure HIP, no MFMA/WMMA. Builds **and runs natively on gfx1201**
(RX 9070 XT, RDNA4). The same source also builds for CDNA via
`--offload-arch=gfx942`.

## Build & run

```bash
./build.sh
```

This compiles with `hipcc -O3 --offload-arch=gfx1201`, greps the ISA for the
wide non-temporal load, then runs the binary.

## Expected output (captured on this gfx1201 box, ROCm 7.2.3)

```
=== verifying wide non-temporal loads in ISA ===
	global_load_b128 v[7:10], v[5:6], off
	global_load_b128 v[13:16], v[4:5], off th:TH_LOAD_NT
	global_load_b128 v[17:20], v[8:9], off th:TH_LOAD_NT
	global_load_b128 v[21:24], v[21:22], off th:TH_LOAD_NT
=== running ===
Device: AMD Radeon RX 9070 XT (gfx1201), 32 CUs, 2400 MHz

Self-check: sum=16777216 ref=16777216 rel_err=0.000e+00 -> PASS

size(MiB)    iters        GB/s
64           50           510
256          50           636
1024         50           637
2048         50           635
```

The RX 9070 XT has ~645 GB/s of GDDR6 peak bandwidth, so ~637 GB/s sustained on
the large (≥256 MiB) working sets is ~99% of peak read bandwidth. The 64 MiB
case is lower because part of it is served from the on-chip cache hierarchy and
the kernel is launch/overhead-bound at that size.

> Numbers vary run-to-run with clocks/thermals; treat the large-size figures as
> the empirical read roofline for this card.

## Files

- `bandwidth_memread.hip` — kernel + self-check + bandwidth sweep (no elisions).
- `build.sh` — exact `hipcc` build, ISA check, and run. Exits 0 on success.
