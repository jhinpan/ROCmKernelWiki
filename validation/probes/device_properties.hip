#include <hip/hip_runtime.h>

#include <cstdio>

static int fail(hipError_t error, const char* operation)
{
    std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
    return 1;
}

int main()
{
    constexpr int device = 0;
    if (hipError_t error = hipSetDevice(device); error != hipSuccess)
        return fail(error, "hipSetDevice(0)");

    hipDeviceProp_t properties{};
    if (hipError_t error = hipGetDeviceProperties(&properties, device);
        error != hipSuccess)
        return fail(error, "hipGetDeviceProperties(0)");

    int runtime_version = 0;
    int driver_version = 0;
    if (hipError_t error = hipRuntimeGetVersion(&runtime_version);
        error != hipSuccess)
        return fail(error, "hipRuntimeGetVersion");
    if (hipError_t error = hipDriverGetVersion(&driver_version);
        error != hipSuccess)
        return fail(error, "hipDriverGetVersion");

    const int waves_per_cu =
        properties.warpSize == 0
            ? 0
            : properties.maxThreadsPerMultiProcessor / properties.warpSize;

    std::printf("device_ordinal=%d\n", device);
    std::printf("name=%s\n", properties.name);
    std::printf("arch=%s\n", properties.gcnArchName);
    std::printf("wavefront_size=%d\n", properties.warpSize);
    std::printf("compute_units=%d\n", properties.multiProcessorCount);
    std::printf("max_threads_per_cu=%d\n",
                properties.maxThreadsPerMultiProcessor);
    std::printf("waves_per_cu=%d\n", waves_per_cu);
    std::printf("shared_mem_per_multiprocessor_bytes=%zu\n",
                properties.sharedMemPerMultiprocessor);
    std::printf("shared_mem_per_block_bytes=%zu\n",
                properties.sharedMemPerBlock);
    std::printf("runtime_version=%d\n", runtime_version);
    std::printf("driver_version=%d\n", driver_version);
    return 0;
}
