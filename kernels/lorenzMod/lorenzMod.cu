#include "main.h"
#include "lorenzMod.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { sigma, rho, beta, kappa };
    enum maps { LLE };
}

__global__ void kernelProgram_lorenzMod(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_lorenzMod);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_lorenzMod(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_lorenzMod, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_lorenzMod(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    numb dx = P(sigma) * (V(y) - V(x))
		+ P(kappa) * sinf(V(y) / 5) * sinf(V(z) / 5);
    numb dy = -V(x) * V(z) + P(rho) * V(x) - V(y)
		+ P(kappa) * sinf(V(x) / 5) * sinf(V(z) / 5);
    numb dz = V(x) * V(y) - P(beta) * V(z)
		+ P(kappa) * cosf(V(y) / 5) * cosf(V(x) / 5);

    Vnext(x) = V(x) + h * dx;
    Vnext(y) = V(y) + h * dy;
    Vnext(z) = V(z) + h * dz;
}
