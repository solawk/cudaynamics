#include "standard_map.h"

#define name standard_map

namespace attributes
{
    enum variables { p, theta };
    enum parameters { K, method, COUNT };
}

__global__ void gpu_wrapper_(name)(Computation* data, uint64_t variation)
{
    kernelProgram_(name)(data, (blockIdx.x * blockDim.x) + threadIdx.x);
}

__host__ __device__ void kernelProgram_(name)(Computation* data, uint64_t variation)
{
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    uint64_t stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(false);
    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_(name));

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_(name)(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis
    AnalysisLobby(data, &finiteDifferenceScheme_(name), variation);
}

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters, PerThread* pt)
{
    V(p) = fmod(V(p), 2.0 * 3.141592);
    V(theta) = fmod(V(theta), 2.0 * 3.141592);
    numb pNext = V(p) + P(K) * sin(V(theta));
    Vnext(p) = pNext;
    Vnext(theta) = V(theta) + pNext;
}

#undef name