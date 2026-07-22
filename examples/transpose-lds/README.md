# transpose-lds — LDS-staged bank-conflict-free matrix transpose (portable HIP)

Out-of-place fp32 transpose `out[x][y] = in[y][x]` staged through LDS. A
`TILE x TILE` (32x32) block is loaded with a coalesced row-major global read,
transposed inside LDS, then written back with a coalesced row-major global
write. Padding is selected for this exact `block(32,32)` lane mapping: the LDS
tile is `[TILE][TILE+1]` on gfx942, while gfx950 uses `[TILE][TILE+2]`
because one wave64 b32 phase contains two adjacent columns.
The two columns then map to its even and odd banks instead of overlapping.

## Classification

**PORTABLE** — pure HIP (LDS + `__syncthreads()`, no MFMA/WMMA intrinsics).
Builds and runs on gfx950; the guarded source also device-compiles for gfx942.

## Build & run

```bash
./build.sh
# or:
hipcc --offload-arch=gfx950 -O3 transpose_lds.cpp -o transpose_lds && ./transpose_lds
```

Optional args: `./transpose_lds <rows> <cols>` (default 2048 x 4096).

## Expected output (captured on MI355X / gfx950)

```
build: OK
Transpose 2048 x 4096 (fp32), TILE=32
avg kernel time: 0.015 ms   effective BW: 4330.7 GB/s
max abs error: 0   mismatches: 0
PASS
```

The transpose is a pure permutation of elements, so the self-check requires
**exact** equality (`max abs error == 0`) against the CPU reference
`out[c*rows + r] == in[r*cols + c]`.

The timing is from one mid-size matrix and is not a tuned peak-bandwidth claim.

## Runs on vs cross-compiles for

- **Runs:** MI355X/gfx950 (`max abs error: 0`).
- **Also compiles for:** gfx942 device code; no gfx942 runtime is claimed.
