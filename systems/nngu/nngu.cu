#include "main.h"
#include "nngu.h"

namespace attributes
{
    enum variables { x, cosx, y, z };
    enum parameters { gamma, eps1, eps2, stepsize };
    enum maps { LLE };
}

__global__ void kernelProgram_nngu(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_nngu);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_nngu(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_nngu, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_nngu(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    numb xmp = fmodf(V(x) + P(stepsize) * 0.5 * V(y), 2.0f * 3.141592f);
    numb ymp = V(y) + P(stepsize) * 0.5 * V(z);
    numb zmp = V(z) + P(stepsize) * 0.5 * ((1 / (P(eps1) * P(eps2))) * (P(gamma) - (P(eps1) + P(eps2)) * V(z) - (1 - (P(eps1) * cosf(V(x)))) * V(y)));

    Vnext(x) = fmodf(V(x) + P(stepsize) * ymp, 2.0f * 3.141592f);
    Vnext(cosx) = cosf(Vnext(x));
    Vnext(y) = V(y) + P(stepsize) * zmp;
    Vnext(z) = V(z) + P(stepsize) * 0.5 * ((1 / (P(eps1) * P(eps2))) * (P(gamma) - (P(eps1) + P(eps2)) * zmp - (1 - (P(eps1) * cosf(xmp))) * ymp));
}