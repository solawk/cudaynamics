#include "main.h"
#include "wilson.h"

namespace attributes
{
    enum variables { v, R, I, T };
    enum parameters { C, tau, p0, p1, p2, p3, p4, p5, p6, p7, Imax, Idc };
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
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    /*if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_wilson);
    }*/
}

__device__ void finiteDifferenceScheme_wilson(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    Vnext(I) = fmodf(V(T), P(Idc)) < (0.5f * P(Idc)) ? P(Imax) : 0.0f;
    Vnext(T) = V(T) + h;

    numb dv = (-(P(p0)+P(p1)*V(v) + P(p2)*V(v)*V(v))*(V(v) - P(p3)) - P(p5)*V(R)*(V(v)-P(p4)) + Vnext(I)) / P(C);
    numb dr = (1.0 / P(tau)) * (-V(R) + P(p6) * V(v) + P(p7));

    //y[0] = x[0] + h*( (-(p[3]+p[4]*x[0]+p[5]*x[0]**2)*(x[0]-p[6])-p[8]*x[1]*(x[0]-p[7]) + sl)/p[1] );
    //y[1] = x[1] + h * ((1 / p[2]) * (-x[1] + p[9] * x[0] + p[10]));

    Vnext(v) = V(v) + h * dv;
    Vnext(R) = V(R) + h * dr;
}