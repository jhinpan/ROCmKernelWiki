# Fused MoE — portable HIP reference (fp32)

A self-checking, **portable** HIP implementation of a Mixture-of-Experts FFN
layer that fuses the whole per-token path into a single kernel:

1. **router GEMV** — `logits[e] = X · Wrouter[e]`
2. **top-k gating** — pick `top_k` experts, softmax over the selected logits
3. **fused expert MLP** (no HBM round-trip for the intermediate `h`):
   - gate-up GEMV: `gate = X·Wgate[e]`, `up = X·Wup[e]`
   - activation: `h = SiLU(gate) * up`  (kept in LDS)
   - down GEMV: `y += gate_weight_e * (h · Wdown[e])`
4. write `y` to the token's output row

One thread-block processes one token end to end; `h` stays in `__shared__`
memory between the gate-up and down projections, which is the whole point of the
fusion. This matches the gate-up + SiLU + down structure described on the
[fused-moe wiki page](../../wiki/kernels/fused-moe.md), but in pure fp32 HIP so
it is portable and numerically verifiable.

## What this is and isn't

- **Is:** a correctness reference showing the fused MoE dataflow and gating math
  that runs and self-checks on RDNA4 (gfx1201).
- **Isn't:** the production kernel. Real MoE replaces the per-token GEMVs with
  **grouped GEMM** (token rows sorted/grouped by expert, padded to `BLOCK_M`)
  executed on **MFMA/WMMA matrix cores** with **FP8** weights and per-block
  dequant scales. Here every token does independent GEMVs in fp32 for clarity.

## Architecture

- **Runs on:** gfx1201 (this box, RDNA4) — and any HIP GPU; it is pure HIP with
  no arch-specific intrinsics.
- Verifies the GPU result against a CPU reference and prints PASS/FAIL + max
  abs/rel error and per-iteration kernel time.

## Build & run

```bash
./build.sh
# or manually:
hipcc --offload-arch=gfx1201 -O3 fused_moe.cpp -o fused_moe && ./fused_moe
```

## Expected output (captured on gfx1201, ROCm 7.2.3)

```
Build OK. Running...
Fused MoE (fp32, portable HIP)
  dims: T=64 D=128 N=256 E=8 top_k=2
  kernel time: 148.223 us/iter (200 iters)
  max abs err: 5.215e-08
  max rel err: 1.508e-03
PASS
```

`max abs err ~5e-8` confirms the GPU fused path matches the CPU reference to
fp32 rounding. (Per-iteration time is for the tiny demo dims and is not a
throughput benchmark.)
