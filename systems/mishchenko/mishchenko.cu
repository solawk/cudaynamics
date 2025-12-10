#include "mishchenko.h"

#define name mishchenko

namespace attributes
{
    enum variables { x, cosx, y, z };
    enum parameters { gamma, eps1, eps2, COUNT };
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

    // Custom area (usually) starts here

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

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters)
{
    numb xmp = fmodf(V(x) + H * 0.5 * V(y), 2.0f * 3.141592f);
    numb ymp = V(y) + H * 0.5 * V(z);
    numb zmp = V(z) + H * 0.5 * ((1 / (P(eps1) * P(eps2))) * (P(gamma) - (P(eps1) + P(eps2)) * V(z) - (1 - (P(eps1) * cosf(V(x)))) * V(y)));

    Vnext(x) = fmodf(V(x) + H * ymp, 2.0f * 3.141592f);
    Vnext(cosx) = cosf(Vnext(x));
    Vnext(y) = V(y) + H * zmp;
    Vnext(z) = V(z) + H * 0.5 * ((1 / (P(eps1) * P(eps2))) * (P(gamma) - (P(eps1) + P(eps2)) * zmp - (1 - (P(eps1) * cosf(xmp))) * ymp));
}

#undef name