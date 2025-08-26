#include "main.h"
#include "izhikevich.h"

namespace attributes
{
    enum variables { v, u, I, t };
    enum parameters { a, b, c, d, p0, p1, p2, p3, Imax, Idc, stepsize, method };
    enum methods { ExplicitEuler };
    enum maps { LLE };
}

__global__ void kernelProgram_izhikevich(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_izhikevich);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_izhikevich(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]));
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_izhikevich, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_izhikevich(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(I) = fmodf(V(t), P(Idc)) < (0.5f * P(Idc)) ? P(Imax) : 0.0f;
        Vnext(t) = V(t) + P(stepsize);
        Vnext(v) = V(v) + P(stepsize) * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(I));
        Vnext(u) = V(u) + P(stepsize) * (P(a) * (P(b) * V(v) - V(u)));

        if (Vnext(v) >= P(p3))
        {
            Vnext(v) = P(c);
            Vnext(u) = Vnext(u) + P(d);
        }
    }
}