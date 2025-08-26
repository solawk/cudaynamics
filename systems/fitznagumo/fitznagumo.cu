#include "main.h"
#include "fitznagumo.h"

namespace attributes
{
    enum variables { v, w };
    enum parameters { a, b, tau, R, Iext, stepsize, method };
    enum methods { ExplicitEuler, ExplicitMidpoint };
    enum maps { LLE };
}

__global__ void kernelProgram_fitznagumo(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_fitznagumo);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_fitznagumo(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_fitznagumo, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_fitznagumo(numb* currentV, numb* nextV, numb* parameters, numb h)
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