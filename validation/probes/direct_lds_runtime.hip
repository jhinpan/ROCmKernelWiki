#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

struct alignas(16) Packet {
    std::uint32_t words[4];
};

__host__ __device__ constexpr std::uint32_t payload_value(unsigned int packet,
                                                           unsigned int word)
{
    return 0x10000000u + packet * 16u + word;
}

__host__ __device__ constexpr std::uint32_t lower_sentinel_value(unsigned int word)
{
    return 0xa11ce000u + word;
}

__host__ __device__ constexpr std::uint32_t upper_sentinel_value(unsigned int word)
{
    return 0xb0b00000u + word;
}

extern "C" __global__ void direct_lds_runtime_probe(Packet* source,
                                                     Packet* output)
{
    __shared__ __align__(16) Packet tile[66];
    const unsigned int lane = threadIdx.x & 63u;

    if (lane == 0) {
        for (unsigned int word = 0; word < 4; ++word) {
            tile[0].words[word] = lower_sentinel_value(word);
            tile[65].words[word] = upper_sentinel_value(word);
        }
    }
    __syncthreads();

    // Every lane supplies its own global source address, while every lane
    // supplies the same LDS base. The 16-byte operation adds 16 * lane to that
    // base; tile[0] and tile[65] detect writes outside the intended interval.
    auto* const lds_base =
        (__attribute__((address_space(3))) Packet*)(tile + 1);
    __builtin_amdgcn_load_to_lds(source + lane, lds_base, 16, 0, 0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();

    output[lane] = tile[lane + 1];
    if (lane == 0) {
        output[64] = tile[0];
        output[65] = tile[65];
    }
}

static bool hip_ok(hipError_t error, const char* operation)
{
    if (error == hipSuccess)
        return true;
    std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
    return false;
}

int main()
{
    constexpr int device = 0;
    constexpr unsigned int packets = 64;
    if (!hip_ok(hipSetDevice(device), "hipSetDevice(0)"))
        return 2;

    std::vector<Packet> source(packets);
    std::vector<Packet> output(packets + 2);
    for (unsigned int packet = 0; packet < packets; ++packet)
        for (unsigned int word = 0; word < 4; ++word)
            source[packet].words[word] = payload_value(packet, word);

    Packet* device_source = nullptr;
    Packet* device_output = nullptr;
    if (!hip_ok(hipMalloc(&device_source, source.size() * sizeof(Packet)),
                "hipMalloc(source)") ||
        !hip_ok(hipMalloc(&device_output, output.size() * sizeof(Packet)),
                "hipMalloc(output)"))
        return 2;

    if (!hip_ok(hipMemcpy(device_source,
                          source.data(),
                          source.size() * sizeof(Packet),
                          hipMemcpyHostToDevice),
                "hipMemcpy(source)"))
        return 2;

    hipLaunchKernelGGL(direct_lds_runtime_probe,
                       dim3(1),
                       dim3(packets),
                       0,
                       nullptr,
                       device_source,
                       device_output);
    if (!hip_ok(hipGetLastError(), "direct_lds_runtime_probe launch") ||
        !hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize") ||
        !hip_ok(hipMemcpy(output.data(),
                          device_output,
                          output.size() * sizeof(Packet),
                          hipMemcpyDeviceToHost),
                "hipMemcpy(output)"))
        return 2;

    unsigned int payload_mismatches = 0;
    unsigned int lower_sentinel_mismatches = 0;
    unsigned int upper_sentinel_mismatches = 0;
    for (unsigned int packet = 0; packet < packets; ++packet)
        for (unsigned int word = 0; word < 4; ++word)
            payload_mismatches +=
                output[packet].words[word] != payload_value(packet, word);
    for (unsigned int word = 0; word < 4; ++word) {
        lower_sentinel_mismatches +=
            output[64].words[word] != lower_sentinel_value(word);
        upper_sentinel_mismatches +=
            output[65].words[word] != upper_sentinel_value(word);
    }

    if (!hip_ok(hipFree(device_output), "hipFree(output)") ||
        !hip_ok(hipFree(device_source), "hipFree(source)"))
        return 2;

    std::printf("device_ordinal=%d\n", device);
    std::printf("copy_width_bytes=%zu\n", sizeof(Packet));
    std::printf("wave_threads=%u\n", packets);
    std::printf("payload_mismatches=%u\n", payload_mismatches);
    std::printf("lower_sentinel_mismatches=%u\n",
                lower_sentinel_mismatches);
    std::printf("upper_sentinel_mismatches=%u\n",
                upper_sentinel_mismatches);

    return payload_mismatches == 0 && lower_sentinel_mismatches == 0 &&
                   upper_sentinel_mismatches == 0
               ? 0
               : 1;
}
