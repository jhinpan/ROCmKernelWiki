# MLA Decode — portable HIP reference (absorbed low-rank latent KV)

A minimal, self-checking **DeepSeek-style Multi-Latent Attention (MLA) decode**
step in pure HIP. It implements the *weight-absorbed* form described in
[`wiki/kernels/mla-decode.md`](../../wiki/kernels/mla-decode.md): the latent
up-projection is folded into Q/output offline, so every query head attends
directly against a single shared **low-rank latent** stream — structurally MQA
with key width `D_C + D_PE` and value width `D_C`.

## What it shows

- The actual decode math AITER's absorbed MLA runs, at `q_len = 1`:
  - score `s = sm_scale · (q_nope·c_kv + q_pe·k_pe)` over the KV history,
  - **online softmax** (running max / denom, FlashAttention-2 style),
  - output accumulates the **latent** value `c_kv` (value == latent).
- One thread block per head; threads cooperatively stream the `N` latent tokens
  and keep the accumulator partitioned across lanes; block-reduction for each
  score; LDS-staged query.
- Shapes are tiny but **realistically proportioned**: latent ≫ rope, value ==
  latent (DeepSeek-V3 uses `D_C=512`, `D_PE=64`; here `D_C=64`, `D_PE=16`).

This is **portable** (pure HIP — FMA, LDS, `__syncthreads`): it builds *and runs*
on this gfx1201 (RDNA4) box and verifies against a double-precision CPU reference.

## Build & run

```bash
./build.sh
```

`build.sh` runs:

```bash
hipcc --offload-arch=gfx1201 -O3 mla_decode.cpp -o mla_decode && ./mla_decode
```

Runs natively on **gfx1201**; pure HIP so it also builds for any other arch by
swapping `--offload-arch`.

## Expected output

```
build: OK
MLA decode (absorbed, low-rank latent KV) -- portable HIP, fp32
  H=16 heads, D_C=64 latent, D_PE=16 rope, N=256 KV tokens
  per-decode: 252.82 us   KV-stream BW: 0.3 GB/s
  max_abs_err = 3.353e-08   max_rel_err = 1.301e-04
PASS
```

`max_abs_err ≈ 3e-8` (fp32 round-off vs the fp64 CPU reference) → **PASS**.

Note the per-decode time and "BW" are **launch-overhead dominated** at these tiny
dims (the KV stream is only ~20 KB), so the GB/s number is *not* a meaningful
bandwidth measurement — it is reported only to illustrate that decode is the
memory-bound term (streaming the KV history). On a real MLA decode you scale
`N`, `H`, `D_C` up and split the sequence (flash-decoding) across the grid, at
which point the kernel approaches HBM roofline. This example optimizes for
*correctness and clarity*, not throughput.
