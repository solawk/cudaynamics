#include "main.h"
#include "halvorsen.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { alpha };
    enum maps { LLE };
}

__global__ void kernelProgram_halvorsen(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step
    int indicesStart = variation * (CUDA_kernel.VAR_COUNT + CUDA_kernel.PARAM_COUNT);   // Start index for the step indices of the attributes in the current variation

    // Custom area (usually) starts here

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_halvorsen(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(0.01f, 50, 0);
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, STEP_INDICES_X(LLE), STEP_INDICES_Y(LLE), &finiteDifferenceScheme_lorenz2);
    }
}

__device__ void finiteDifferenceScheme_halvorsen(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    numb dx = -P(alpha) * V(x) - 4 * V(y) - 4 * V(z) - V(y) * V(y);
    numb dy = -P(alpha) * V(y) - 4 * V(z) - 4 * V(x) - V(z) * V(z);
    numb dz = -P(alpha) * V(z) - 4 * V(x) - 4 * V(y) - V(x) * V(x);

    Vnext(x) = V(x) + h * dx;
    Vnext(y) = V(y) + h * dy;
    Vnext(z) = V(z) + h * dz;
}