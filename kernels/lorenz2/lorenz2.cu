#include "main.h"
#include "lorenz2.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { sigma, rho, beta, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRK4 };
    enum maps { LLE };
}

__global__ void kernelProgram_lorenz2(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation + data->startVariationInCurrentExecute
        >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_lorenz2);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_lorenz2(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_lorenz2);
    }
}

__device__ void finiteDifferenceScheme_lorenz2(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        numb dx = P(sigma) * (V(y) - V(x));
        numb dy = V(x) * (P(rho) - V(z)) - V(y);
        numb dz = V(x) * V(y) - P(beta) * V(z);

        Vnext(x) = V(x) + h * dx;
        Vnext(y) = V(y) + h * dy;
        Vnext(z) = V(z) + h * dz;
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb dx = P(sigma) * (V(y) - V(x));
        numb dy = V(x) * (P(rho) - V(z)) - V(y);
        numb dz = V(x) * V(y) - P(beta) * V(z);

        numb xmp = V(x) + h * 0.5 * dx;
        numb ymp = V(y) + h * 0.5 * dy;
        numb zmp = V(z) + h * 0.5 * dz;

        numb dx2 = P(sigma) * (ymp - xmp);
        numb dy2 = xmp * (P(rho) - zmp) - ymp;
        numb dz2 = xmp * ymp - P(beta) * zmp;

        Vnext(x) = V(x) + h * dx2;
        Vnext(y) = V(y) + h * dy2;
        Vnext(z) = V(z) + h * dz2;
    }
}