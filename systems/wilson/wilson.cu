#include "main.h"
#include "wilson.h"

namespace attributes
{
    enum variables { V, R, I, T };
    enum parameters { C, tau, p0, p1, p2, p3, p4, p5, p6, p7, Imax, Idc, stepsize, method };
    enum methods { ExplicitEuler };
    enum maps { LLE };
}

__global__ void kernelProgram_wilson(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_wilson);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_wilson(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]));
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_wilson, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_wilson(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(I) = fmodf(V(T), P(Idc)) < (0.5f * P(Idc)) ? P(Imax) : 0.0f;
        Vnext(T) = V(T) + P(stepsize);
        Vnext(V) = V(V) + P(stepsize) * ((-(P(p0) + P(p1) * V(V) + P(p2) * V(V) * V(V)) * (V(V) - P(p3)) - P(p5) * V(R) * (V(V) - P(p4)) + Vnext(I)) / P(C));
        Vnext(R) = V(R) + P(stepsize) * ((1.0 / P(tau)) * (-V(R) + P(p6) * V(V) + P(p7)));
    }
}