#include "main.h"
#include "bolshakov.h"

namespace attributes
{
    enum variables { Q, S, X, Y, c };
    enum parameters { i, p, k, r };
    enum maps { LLE };
}

__global__ void kernelProgram_bolshakov(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_bolshakov);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_bolshakov(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_bolshakov, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_bolshakov(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    Vnext(Q) = V(Q) + V(c) * (P(i) + V(S)) - !V(c) * (P(i) + V(S));

    if (Vnext(Q) < 0) Vnext(Q) = 0;
    if (Vnext(Q) > P(r)) Vnext(Q) = P(r);

    Vnext(X) = P(p) * (Vnext(Q) + V(X));
    Vnext(Y) = P(k) * (Vnext(X) - V(Y)) + V(Y);
    Vnext(S) = Vnext(X) - Vnext(Y);
    Vnext(c) = (V(c) || (Vnext(Q) > 0 ? 0 : 1)) && (Vnext(Q) < P(r) ? 1 : 0);
}