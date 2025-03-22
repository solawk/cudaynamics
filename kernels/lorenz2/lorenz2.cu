#include "main.h"
#include "lorenz2.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { sigma, rho, beta };
}

__global__ void kernelProgram_lorenz2(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step
    int indicesStart = variation * (CUDA_kernel.VAR_COUNT + CUDA_kernel.PARAM_COUNT);

    // Custom area (usually) starts here

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_lorenz2(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    LLE(data, variation,
        CUDA_marshal.stepIndices[indicesStart + (CUDA_kernel.mapDatas[0].typeX == PARAMETER ? CUDA_kernel.VAR_COUNT : 0) + CUDA_kernel.mapDatas[0].indexX],
        CUDA_marshal.stepIndices[indicesStart + (CUDA_kernel.mapDatas[0].typeY == PARAMETER ? CUDA_kernel.VAR_COUNT : 0) + CUDA_kernel.mapDatas[0].indexY],
        &finiteDifferenceScheme_lorenz2);
}

__device__ void finiteDifferenceScheme_lorenz2(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    numb dx = P(sigma) * (V(y) - V(x));
    numb dy = V(x) * (P(rho) - V(z)) - V(y);
    numb dz = V(x) * V(y) - P(beta) * V(z);

    Vnext(x) = V(x) + h * dx;
    Vnext(y) = V(y) + h * dy;
    Vnext(z) = V(z) + h * dz;
}