#include "main.h"
#include "rlcs_jj.h"

namespace attributes
{
    enum variables { x1, sin_x1, x2, x3 };
    enum parameters { p0, p1, p2, p3, p4, p5 };
    enum maps { LLE };
}

__global__ void kernelProgram_rlcs_jj(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step
    //int indicesStart = variation * (CUDA_kernel.VAR_COUNT + CUDA_kernel.PARAM_COUNT);   // Start index for the step indices of the attributes in the current variation

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_rlcs_jj);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_rlcs_jj(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_rlcs_jj, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_rlcs_jj(numb* currentV, numb* nextV, numb* parameters, numb h)
{
	Vnext(x1) = fmodf(V(x1) + h * V(x2), 2.0f * 3.141592f);
	Vnext(sin_x1) = sinf(Vnext(x1));
	Vnext(x3) = V(x3) + h * (1.0f / P(p0)) * (V(x2) - V(x3));
	Vnext(x2) = V(x2) + h * (1.0f / P(p1)) *
        (
            P(p2)
            - ((V(x2) > P(p3)) ? P(p4) : P(p5)) * V(x2)
            - sinf(Vnext(x1))
            - Vnext(x3)
            );
}
