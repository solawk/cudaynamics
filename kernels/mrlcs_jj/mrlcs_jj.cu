#include "main.h"
#include "mrlcs_jj.h"

namespace attributes
{
    enum variables { sin_x0, x0, x0true, x1, x2 };
    enum parameters { p0, p1, p2, p3, p4, p5, p6 };
    enum maps { LLE };
}

__global__ void kernelProgram_mrlcs_jj(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step
    //int indicesStart = variation * (CUDA_kernel.VAR_COUNT + CUDA_kernel.PARAM_COUNT);   // Start index for the step indices of the attributes in the current variation

    // Custom area (usually) starts here

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_mrlcs_jj(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(0.01f, 50, 0);
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_mrlcs_jj);
    }
}

__device__ void finiteDifferenceScheme_mrlcs_jj(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    numb hs = (1.0f - P(p6)) * h;

    Vnext(x0) = fmodf(V(x0) + hs * V(x1), 2.0f * 3.141592f);
    Vnext(sin_x0) = sinf(Vnext(x0));
    Vnext(x0true) = V(x0true) + hs * V(x1);
    Vnext(x2) = V(x2) + hs * ((1.0f / P(p0)) * (V(x1) - V(x2)));
    Vnext(x1) = V(x1) + hs * ((1.0f / P(p1)) * (P(p3) - P(p2) * (1 + P(p4) * cosf(Vnext(x0))) * V(x1) - sinf(Vnext(x0)) - P(p5) * Vnext(x2)));

    hs = P(p6) * h;
    V(x0) = Vnext(x0);
    V(sin_x0) = sinf(V(x0));
    V(x1) = Vnext(x1);
    V(x2) = Vnext(x2);

    Vnext(x0) = fmodf(V(x0) + hs * V(x1), 2.0f * 3.141592f);
    Vnext(sin_x0) = sinf(Vnext(x0));
    Vnext(x0true) = V(x0true) + hs * V(x1);
    Vnext(x2) = V(x2) + hs * (1.0f / P(p0)) * (V(x1) - V(x2)) / (1.0f + hs * (1.0f / P(p0)));
    Vnext(x1) = V(x1) + hs * ((1.0f / P(p1)) * (P(p3) - P(p2) * (1 + P(p4) * cosf(Vnext(x0))) * V(x1) - sinf(Vnext(x0)) - P(p5) * Vnext(x2)));
    Vnext(x1) = V(x1) + hs * ((1.0f / P(p1)) * (P(p3) - P(p2) * (1 + P(p4) * cosf(Vnext(x0))) * Vnext(x1) - sinf(Vnext(x0)) - P(p5) * Vnext(x2)));
}