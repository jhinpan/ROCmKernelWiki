// fp8_gemm_cdna.cpp — CDNA-MFMA FP8 GEMM via the f8f6f4 matrix-core path.
//
// COMPILER/ISA-VERIFY ONLY. These intrinsics map to v_mfma_*_f8f6f4 (gfx950)
// and v_mfma_*_fp8_fp8 (gfx942). build.sh compiles both targets and checks the
// emitted ISA; it does not launch either kernel.
//
// Two regimes, both real and selected at compile time by --offload-arch:
//
//  * gfx950 (CDNA4): OCP-compliant E4M3, unified f8f6f4 MMA with K=128 and
//    HARDWARE block scaling. The only builtin the toolchain exposes is the
//    *scaled* form __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4; pass E8M0
//    scale operands of 0 (==2^0==1.0, since E8M0 is biased exponent 127 -> the
//    encoded byte 127 means 1.0; the intrinsic takes the raw exponent and 0
//    disables scaling when ABID/opsel are clear) for a plain dense GEMM, or real
//    per-32-K-block E8M0 exponents for MXFP8.
//
//  * gfx942 (CDNA3): FNUZ-encoded FP8 (__hip_fp8_e4m3_fnuz), narrower MMA with
//    K=32 (v_mfma_f32_16x16x32_fp8_fp8) and NO hardware MX scaling — block
//    scales must be applied in software. FNUZ != OCP: the bit patterns differ
//    (FNUZ has no signed zero / Inf, one extra exponent value), so FP8 weights
//    must be re-encoded when moving gfx942 <-> gfx950. See the migration note.
//
// Build:  see build.sh  (produces gfx950 object+exe and a gfx942 object).

#include <hip/hip_runtime.h>
#include <cstdint>

using fp8x32 = __attribute__((__vector_size__(32))) unsigned char; // 32xE4M3 bytes
using f32x4  = __attribute__((__vector_size__(16))) float;         // 4 FP32 accum
using f32x16 = __attribute__((__vector_size__(64))) float;         // 16 FP32 accum

// ---------------------------------------------------------------------------
// gfx950 (CDNA4) — OCP FP8, unified f8f6f4 path, K=128.
// ---------------------------------------------------------------------------
#if defined(__gfx950__)

// One wave64 computes a 16x16 FP32 output tile from a 16x128 A slab and a
// 128x16 B slab of OCP E4M3 (CBSZ/BLGP=000). scaleA/scaleB are packed E8M0
// exponents (one per 32-element K block); pass 0 to disable scaling (dense GEMM)
// or the real exponents for MXFP8.
__device__ inline f32x4 mma_16x16x128_f8(fp8x32 a, fp8x32 b, f32x4 acc,
                                         int scaleA, int scaleB)
{
    // args: a, b, acc, cbsz(A fmt=E4M3=0), blgp(B fmt=E4M3=0),
    //       opselA(scale-byte select=0), scaleA, opselB=0, scaleB
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, acc, /*cbsz=*/0, /*blgp=*/0,
        /*opselA=*/0, scaleA, /*opselB=*/0, scaleB);
}

// Minimal whole-GEMM kernel over K (multiple of 128). A row-major (MxK),
// B col-major (KxN). Each block/wave handles one 16x16 C tile. This proves the
// f8f6f4 instruction is reachable from a complete kernel; production code adds
// LDS staging, double buffering and the 4-wave schedule.
extern "C" __global__ void fp8_gemm_gfx950(const fp8x32* __restrict__ A,
                                           const fp8x32* __restrict__ B,
                                           float* __restrict__ C,
                                           const int* __restrict__ scaleA,
                                           const int* __restrict__ scaleB,
                                           int M, int N, int K)
{
    int tileM = blockIdx.y, tileN = blockIdx.x;
    int kblocks = K / 128;                 // each fp8x32 packs 32 K-elems; lane spread gives 128
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    int lane = threadIdx.x;                // 0..63 wave64
    for (int kb = 0; kb < kblocks; ++kb) {
        fp8x32 a = A[(tileM * kblocks + kb) * 64 + lane];
        fp8x32 b = B[(tileN * kblocks + kb) * 64 + lane];
        int sA = scaleA ? scaleA[tileM * kblocks + kb] : 0;
        int sB = scaleB ? scaleB[tileN * kblocks + kb] : 0;
        acc = mma_16x16x128_f8(a, b, acc, sA, sB);
    }
    // Store 4 accumulator regs (layout is wave-specific; sketch only).
    int row = tileM * 16 + (lane / 16);
    int col = tileN * 16 + (lane % 16);
    if (row < M && col < N)
        for (int i = 0; i < 4; ++i)
            C[(row + i * 4) * N + col] = acc[i];
}

#endif // __gfx950__

// ---------------------------------------------------------------------------
// gfx942 (CDNA3) — FNUZ FP8, K=32 MMA, software scaling.
// ---------------------------------------------------------------------------
#if defined(__gfx942__)

// Wave64 16x16x32 FNUZ FP8 MMA. Each operand register holds 8 bytes (a long)
// = 8 E4M3-FNUZ values per lane; 64 lanes * 8 = 512 = 16*32 elements.
__device__ inline f32x4 mma_16x16x32_fp8_fnuz(long a, long b, f32x4 acc)
{
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc, 0, 0, 0);
}

extern "C" __global__ void fp8_gemm_gfx942(const long* __restrict__ A,
                                           const long* __restrict__ B,
                                           float* __restrict__ C,
                                           const float* __restrict__ blockScaleA,
                                           const float* __restrict__ blockScaleB,
                                           int M, int N, int K)
{
    int tileM = blockIdx.y, tileN = blockIdx.x;
    int kblocks = K / 32;
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    int lane = threadIdx.x;
    for (int kb = 0; kb < kblocks; ++kb) {
        long a = A[(tileM * kblocks + kb) * 64 + lane];
        long b = B[(tileN * kblocks + kb) * 64 + lane];
        f32x4 part = mma_16x16x32_fp8_fnuz(a, b, (f32x4){0,0,0,0});
        // NO hardware MX on CDNA3: dequantize the block in software.
        float s = blockScaleA[tileM * kblocks + kb] * blockScaleB[tileN * kblocks + kb];
        for (int i = 0; i < 4; ++i) acc[i] += part[i] * s;
    }
    int row = tileM * 16 + (lane / 16);
    int col = tileN * 16 + (lane % 16);
    if (row < M && col < N)
        for (int i = 0; i < 4; ++i)
            C[(row + i * 4) * N + col] = acc[i];
}

#endif // __gfx942__

// Host entry so the file links into an executable. It does not launch a kernel;
// it only proves the translation unit and device code object link end to end.
int main(int argc, char**)
{
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return 0;
    return argc > 100000 ? dev : 0; // never executes MMA here
}
