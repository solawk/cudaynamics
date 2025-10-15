#include "main.h"
#include "mishchenko.h"

namespace attributes
{
    enum variables { x, cosx, y, z };
    enum parameters { gamma, eps1, eps2, stepsize };
    enum maps { LLE, MAX, Period};
}

__global__ void kernelProgram_mishchenko(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES;

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
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_mishchenko, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_mishchenko, MO(MAX));
    }

    if (M(Period).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(Period, 1), MS(Period, 2), MS(Period, 3), MS(Period, 4), MS(Period, 5), MS(Period, 6), MS(Period, 7), attributes::parameters::stepsize);
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_mishchenko, MO(Period));
    }
}

__device__ __forceinline__ void finiteDifferenceScheme_mishchenko(numb* currentV, numb* nextV, numb* parameters)
{
    numb xmp = fmodf(V(x) + P(stepsize) * 0.5 * V(y), 2.0f * 3.141592f);
    numb ymp = V(y) + P(stepsize) * 0.5 * V(z);
    numb zmp = V(z) + P(stepsize) * 0.5 * ((1 / (P(eps1) * P(eps2))) * (P(gamma) - (P(eps1) + P(eps2)) * V(z) - (1 - (P(eps1) * cosf(V(x)))) * V(y)));

    Vnext(x) = fmodf(V(x) + P(stepsize) * ymp, 2.0f * 3.141592f);
    Vnext(cosx) = cosf(Vnext(x));
    Vnext(y) = V(y) + P(stepsize) * zmp;
    Vnext(z) = V(z) + P(stepsize) * 0.5 * ((1 / (P(eps1) * P(eps2))) * (P(gamma) - (P(eps1) + P(eps2)) * zmp - (1 - (P(eps1) * cosf(xmp))) * ymp));
}