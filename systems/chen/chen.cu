#include "main.h"
#include "chen.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { alpha, beta, gamma, delta, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE };
}

__global__ void kernelProgram_chen(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_chen);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_chen(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_chen, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_chen(numb* currentV, numb* nextV, numb* parameters, numb h)
{

    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (P(alpha) * V(x) - V(y) * V(z));
        Vnext(y) = V(y) + P(stepsize) * (P(beta) * V(y) + V(x) * V(z));
        Vnext(z) = V(z) + P(stepsize) * (P(delta) * V(z) + V(x) * V(y) / P(gamma));
    }


    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + 0.5 * P(stepsize) * (P(alpha) * V(x) - V(y) * V(z));
        numb ymp = V(y) + 0.5 * P(stepsize) * (P(beta) * V(y) + V(x) * V(z));
        numb zmp = V(z) + 0.5 * P(stepsize) * (P(delta) * V(z) + V(x) * V(y) / P(gamma));

        Vnext(x) = V(x) + P(stepsize) * (P(alpha) * xmp - ymp * zmp);
        Vnext(y) = V(y) + P(stepsize) * (P(beta) * ymp + xmp * zmp);
        Vnext(z) = V(z) + P(stepsize) * (P(delta) * zmp + xmp * ymp / P(gamma));
    }


    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(alpha) * V(x) - V(y) * V(z);
        numb ky1 = P(beta) * V(y) + V(x) * V(z);
        numb kz1 = P(delta) * V(z) + V(x) * V(y) / P(gamma);

        numb xmp = V(x) + 0.5 * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5 * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5 * P(stepsize) * kz1;

        numb kx2 = P(alpha) * xmp - ymp * zmp;
        numb ky2 = P(beta) * ymp + xmp * zmp;
        numb kz2 = P(delta) * zmp + xmp * ymp / P(gamma);

        xmp = V(x) + 0.5 * P(stepsize) * kx2;
        ymp = V(y) + 0.5 * P(stepsize) * ky2;
        zmp = V(z) + 0.5 * P(stepsize) * kz2;

        numb kx3 = P(alpha) * xmp - ymp * zmp;
        numb ky3 = P(beta) * ymp + xmp * zmp;
        numb kz3 = P(delta) * zmp + xmp * ymp / P(gamma);

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = P(alpha) * xmp - ymp * zmp;
        numb ky4 = P(beta) * ymp + xmp * zmp;
        numb kz4 = P(delta) * zmp + xmp * ymp / P(gamma);

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }


    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * P(stepsize) - P(symmetry);
        numb h2 = 0.5 * P(stepsize) + P(symmetry);

        numb xmp = V(x) + h1 * (P(alpha) * V(x) - V(y) * V(z));
        numb ymp = V(y) + h1 * (P(beta) * V(y) + xmp * V(z));
        numb zmp = V(z) + h1 * (P(delta) * V(z) + xmp * ymp / P(gamma));

        Vnext(z) = (zmp + xmp * ymp / P(gamma) * h2) / (1 - P(delta) * h2);
        Vnext(y) = (ymp + xmp * Vnext(z) * h2) / (1 - P(beta) * h2);
        Vnext(x) = (xmp - Vnext(y) * Vnext(z) * h2) / (1 - P(alpha) * h2);      
    }

}