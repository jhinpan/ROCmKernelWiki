# transpose-lds — LDS-staged bank-conflict-free matrix transpose (portable HIP)

Out-of-place fp32 transpose `out[x][y] = in[y][x]` staged through LDS. A
`TILE x TILE` (32x32) block is loaded with a coalesced row-major global read,
transposed inside LDS, then written back with a coalesced row-major global
write. The LDS tile is padded to `[TILE][TILE+1]` so the transposed
column-wise reads land in distinct banks (`gcd(TILE+1, banks) == 1`),
eliminating the 32-way bank conflict the naive unpadded layout would suffer.

## Classification

**PORTABLE** — pure HIP (LDS + `__syncthreads()`, no MFMA/WMMA intrinsics).
Builds and runs natively on this gfx1201 (RDNA4) box; the same source also
compiles for gfx942/gfx950.

## Build & run

```bash
./build.sh
# or:
hipcc --offload-arch=gfx1201 -O3 transpose_lds.cpp -o transpose_lds && ./transpose_lds
```

Optional args: `./transpose_lds <rows> <cols>` (default 2048 x 4096).

## Expected output (captured on gfx1201, RX 9070 XT, ROCm 7.2.3)

```
build: OK
Transpose 2048 x 4096 (fp32), TILE=32
avg kernel time: 0.138 ms   effective BW: 487.2 GB/s
max abs error: 0   mismatches: 0
PASS
```

The transpose is a pure permutation of elements, so the self-check requires
**exact** equality (`max abs error == 0`) against the CPU reference
`out[c*rows + r] == in[r*cols + c]`.

The reported effective bandwidth (`2 * bytes / time`, read + write) is real
gfx1201 timing for a single mid-size matrix and is included for illustration,
not as a tuned peak-bandwidth benchmark.

## Runs on vs cross-compiles for

- **Runs:** gfx1201 (verified here).
- **Also compiles for:** gfx942, gfx950 (pure HIP, no arch-specific intrinsics).
