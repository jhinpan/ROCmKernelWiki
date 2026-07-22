# bandwidth-microbench — portable float4 HBM read bandwidth

Pure-HIP, **runnable** cousin of gcnasm's `bandwidth_memread`. A persistent
grid-stride kernel streams a large array with **128-bit (`float4`) non-temporal
loads** and accumulates into a register sink so the compiler cannot dead-code
the loads away. It sweeps several working-set sizes and reports measured GB/s.

## What it shows

- 128-bit vectorized loads: the gfx950 ISA emits
  `global_load_dwordx4 ... off nt` (verified by `build.sh`).
- Persistent grid (a couple of blocks per CU) + manual `UNROLL=8` so several
  loads are in flight before the first `vmcnt` wait.
- A self-check kernel reduces a known buffer (every scalar = 1.0f) and compares
  the sum against a CPU reference, proving the load path is correct.

## Classification

**PORTABLE** — pure HIP, no matrix instructions. Builds and runs on gfx950; the
captured MI355X run below includes both the ISA and numeric checks.

## Build & run

```bash
./build.sh
```

This compiles with `hipcc -O3 --offload-arch=gfx950`, greps the ISA for the
wide non-temporal load, then runs the binary.

## Expected output (captured on MI355X / gfx950)

```
=== verifying wide non-temporal loads in ISA ===
	global_load_dwordx4 v[8:11], v[6:7], off
	global_load_dwordx4 v[10:13], v[4:5], off nt
	global_load_dwordx4 v[14:17], v[42:43], off nt
	global_load_dwordx4 v[18:21], v[44:45], off nt
=== running ===
Device: AMD Instinct MI355X (gfx950:sramecc+:xnack-), 256 CUs, 2400 MHz

Self-check: sum=16777216 ref=16777216 rel_err=0.000e+00 -> PASS

size(MiB)    iters        GB/s
64           50           6152
256          50           6396
1024         50           6192
2048         50           6192

RESULT: PASS
```

These are the values from one run, not a peak-bandwidth claim.

## Files

- `bandwidth_memread.hip` — kernel + self-check + bandwidth sweep (no elisions).
- `build.sh` — exact `hipcc` build, ISA check, and run. Exits 0 on success.
