# vector-add-asm — portable HIP vadd + GCN inline-asm CDNA variant

Two complementary pieces for `C[i] = A[i] + B[i]` (FP32), the canonical
memory-bound elementwise kernel:

1. **`vadd_hip.cpp`** — **PORTABLE** pure-HIP grid-stride vector add. Builds and
   **runs on gfx1201** (RDNA4, this box). Self-checks against a CPU reference and
   reports effective bandwidth (12 B/element: two reads + one write).
2. **`vadd_asm_gfx942.cpp`** — **CDNA-MFMA-style asm path, CROSS-COMPILE-ONLY.**
   A GCN inline-assembly vector add using `global_load_dword` /
   `global_store_dword` gated by `s_waitcnt vmcnt(0)` — the same VMEM /
   wait-counter mechanics the wiki page's hand-written `buffer_*` kernel uses,
   reduced to a runnable inline block so the assembler validates the encodings.
   Built with `--offload-arch=gfx942`; **not executed on gfx1201**.

## Build & run

```bash
./build.sh
```

`build.sh` builds and runs Part 1 on gfx1201, then cross-compiles Part 2 to an
object file for gfx942 (object-only — proves the GCN asm assembles).

## Expected output (captured on gfx1201, ROCm 7.2.3)

```
=== Part 1: portable HIP vadd (build + RUN on gfx1201) ===
vadd HIP (portable, gfx1201): N=16777216  block=256 grid=4096
  time = 0.350 ms/iter   effective BW = 574.8 GB/s (12 B/elem)
  max abs err = 0
  PASS

=== Part 2: GCN inline-asm vadd (CROSS-COMPILE-ONLY for gfx942) ===
OK: vadd_asm_gfx942.o produced (not executed on gfx1201)
-rw-rw-r-- 1 ... vadd_asm_gfx942.o
```

(Bandwidth and timing vary by run; `max abs err = 0` / `PASS` are deterministic
because the data is integer-derived and the result is exact.)

## Verifying the GCN asm actually assembled

The device ISA for the gfx942 target contains the inline VMEM ops:

```bash
hipcc --offload-arch=gfx942 -O3 -S vadd_asm_gfx942.cpp -o vadd_asm.s
grep -E "global_load_dword|global_store_dword|v_add_f32" vadd_asm.s
#   global_load_dword v6, v[0:1], off
#   global_load_dword v7, v[4:5], off
#   v_add_f32 v0, v6, v7
#   global_store_dword v[2:3], v0, off
```

## Which arch runs vs cross-compiles

| File                   | gfx1201 (this box) | gfx942 (MI300) |
|------------------------|--------------------|----------------|
| `vadd_hip.cpp`         | builds + **runs**  | builds + runs  |
| `vadd_asm_gfx942.cpp`  | n/a                | builds (object); runs on CDNA hardware |

## Notes

- The portable HIP kernel is what you'd ship in practice — the compiler emits the
  loads/stores and schedules `s_waitcnt`. The inline-asm file exists to make the
  VMEM + `s_waitcnt vmcnt` path explicit, mirroring the page's annotated
  `buffer_load_dword ... lds` / double-buffer kernel.
- `global_*` ops take a 64-bit address in a VGPR pair and use the VMEM (`vmcnt`)
  counter, so a single `s_waitcnt vmcnt(0)` drains the two loads before the add.
