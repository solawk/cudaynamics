#include "main.h"
#include "mrlcs_jj.h"

namespace attributes
{
    enum variables { sin_theta, theta, theta_true, Vvar, IL };
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

    TRANSIENT_SKIP(finiteDifferenceScheme_mrlcs_jj);

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
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_mrlcs_jj);
    }
}

__device__ void finiteDifferenceScheme_mrlcs_jj(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    numb hs = (1.0f - P(p6)) * h;

    Vnext(theta) = fmodf(V(theta) + hs * V(Vvar), 2.0f * 3.141592f);
    Vnext(sin_theta) = sinf(Vnext(theta));
    Vnext(theta_true) = V(theta_true) + hs * V(Vvar);
    Vnext(IL) = V(IL) + hs * ((1.0f / P(p0)) * (V(Vvar) - V(IL)));
    Vnext(Vvar) = V(Vvar) + hs * ((1.0f / P(p1)) * (P(p3) - P(p2) * (1 + P(p4) * cosf(Vnext(theta))) * V(Vvar) - sinf(Vnext(theta)) - P(p5) * Vnext(IL)));

    hs = P(p6) * h;
    V(theta) = Vnext(theta);
    V(sin_theta) = sinf(V(theta));
    V(Vvar) = Vnext(Vvar);
    V(IL) = Vnext(IL);

    Vnext(theta) = fmodf(V(theta) + hs * V(Vvar), 2.0f * 3.141592f);
    Vnext(sin_theta) = sinf(Vnext(theta));
    Vnext(theta_true) = V(theta_true) + hs * V(Vvar);
    Vnext(IL) = V(IL) + hs * (1.0f / P(p0)) * (V(Vvar) - V(IL)) / (1.0f + hs * (1.0f / P(p0)));
    Vnext(Vvar) = V(Vvar) + hs * ((1.0f / P(p1)) * (P(p3) - P(p2) * (1 + P(p4) * cosf(Vnext(theta))) * V(Vvar) - sinf(Vnext(theta)) - P(p5) * Vnext(IL)));
    Vnext(Vvar) = V(Vvar) + hs * ((1.0f / P(p1)) * (P(p3) - P(p2) * (1 + P(p4) * cosf(Vnext(theta))) * Vnext(Vvar) - sinf(Vnext(theta)) - P(p5) * Vnext(IL)));
}