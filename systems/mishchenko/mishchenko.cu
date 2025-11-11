#include "main.h"
#include "mishchenko.h"

namespace attributes
{
    enum variables { x, cosx, y, z };
    enum parameters { gamma, eps1, eps2, COUNT };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_mishchenko(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_mishchenko);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_mishchenko(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE(data, variation, &finiteDifferenceScheme_mishchenko);
    }

    if (M(MAX).toCompute)
    {
        MAX(data, variation, &finiteDifferenceScheme_mishchenko);
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        Period(data, variation, &finiteDifferenceScheme_mishchenko);
    }
}

__device__ __forceinline__ void finiteDifferenceScheme_mishchenko(numb* currentV, numb* nextV, numb* parameters)
{
    numb xmp = fmodf(V(x) + H * 0.5 * V(y), 2.0f * 3.141592f);
    numb ymp = V(y) + H * 0.5 * V(z);
    numb zmp = V(z) + H * 0.5 * ((1 / (P(eps1) * P(eps2))) * (P(gamma) - (P(eps1) + P(eps2)) * V(z) - (1 - (P(eps1) * cosf(V(x)))) * V(y)));

    Vnext(x) = fmodf(V(x) + H * ymp, 2.0f * 3.141592f);
    Vnext(cosx) = cosf(Vnext(x));
    Vnext(y) = V(y) + H * zmp;
    Vnext(z) = V(z) + H * 0.5 * ((1 / (P(eps1) * P(eps2))) * (P(gamma) - (P(eps1) + P(eps2)) * zmp - (1 - (P(eps1) * cosf(xmp))) * ymp));
}