#include "main.h"
#include "fitzhugh_nagumo.h"

namespace attributes
{
    enum variables { v, w };
    enum parameters { a, b, tau, R, Iext, stepsize, method };
    enum methods { ExplicitEuler, ExplicitMidpoint };
    enum maps { LLE };
}

__global__ void kernelProgram_fitzhugh_nagumo(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_fitzhugh_nagumo);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_fitzhugh_nagumo(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]));
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_fitzhugh_nagumo, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_fitzhugh_nagumo(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(v) = V(v) + P(stepsize) * (V(v) - (V(v) * V(v) * V(v) / 3.0f) - V(w) + P(R) * P(Iext));
        Vnext(w) = V(w) + P(stepsize) * ((V(v) + P(a) - P(b) * V(w)) / P(tau));
    }
    
    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb vmp = V(v) + P(stepsize) * 0.5f * (V(v) - (V(v) * V(v) * V(v) / 3.0f) - V(w) + P(R) * P(Iext));
        numb wmp = V(w) + P(stepsize) * 0.5f * ((V(v) + P(a) - P(b) * V(w)) / P(tau));

        Vnext(v) = V(v) + P(stepsize) * (vmp - (vmp * vmp * vmp / 3.0f) - wmp + P(R) * P(Iext));
        Vnext(w) = V(w) + P(stepsize) * ((vmp + P(a) - P(b) * wmp) / P(tau));
    }
}