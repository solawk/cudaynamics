#include "main.h"
#include "halvorsen.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { alpha, beta, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE };
}

__global__ void kernelProgram_halvorsen(Computation* data)
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

        finiteDifferenceScheme_halvorsen(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(0.01f, 50, 0);
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_halvorsen, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_halvorsen(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (-P(alpha) * V(x) - P(beta) * V(y) - P(beta) * V(z) - V(y) * V(y));
        Vnext(y) = V(y) + P(stepsize) * (-P(alpha) * V(y) - P(beta) * V(z) - P(beta) * V(x) - V(z) * V(z));
        Vnext(z) = V(z) + P(stepsize) * (-P(alpha) * V(z) - P(beta) * V(x) - P(beta) * V(y) - V(x) * V(x));
    }


    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + 0.5 * P(stepsize) * (-P(alpha) * V(x) - P(beta) * V(y) - P(beta) * V(z) - V(y) * V(y));
        numb ymp = V(y) + 0.5 * P(stepsize) * (-P(alpha) * V(y) - P(beta) * V(z) - P(beta) * V(x) - V(z) * V(z));
        numb zmp = V(z) + 0.5 * P(stepsize) * (-P(alpha) * V(z) - P(beta) * V(x) - P(beta) * V(y) - V(x) * V(x));

        Vnext(x) = V(x) + P(stepsize) * (-P(alpha) * xmp - P(beta) * ymp - P(beta) * zmp - ymp * ymp);
        Vnext(y) = V(y) + P(stepsize) * (-P(alpha) * ymp - P(beta) * zmp - P(beta) * xmp - zmp * zmp);
        Vnext(z) = V(z) + P(stepsize) * (-P(alpha) * zmp - P(beta) * xmp - P(beta) * ymp - xmp * xmp);
    }


    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = -P(alpha) * V(x) - P(beta) * V(y) - P(beta) * V(z) - V(y) * V(y);
        numb ky1 = -P(alpha) * V(y) - P(beta) * V(z) - P(beta) * V(x) - V(z) * V(z);
        numb kz1 = -P(alpha) * V(z) - P(beta) * V(x) - P(beta) * V(y) - V(x) * V(x);

        numb xmp = V(x) + 0.5 * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5 * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5 * P(stepsize) * kz1;

        numb kx2 = -P(alpha) * xmp - P(beta) * ymp - P(beta) * zmp - ymp * ymp;
        numb ky2 = -P(alpha) * ymp - P(beta) * zmp - P(beta) * xmp - zmp * zmp;
        numb kz2 = -P(alpha) * zmp - P(beta) * xmp - P(beta) * ymp - xmp * xmp;

        xmp = V(x) + 0.5 * P(stepsize) * kx2;
        ymp = V(y) + 0.5 * P(stepsize) * ky2;
        zmp = V(z) + 0.5 * P(stepsize) * kz2;

        numb kx3 = -P(alpha) * xmp - P(beta) * ymp - P(beta) * zmp - ymp * ymp;
        numb ky3 = -P(alpha) * ymp - P(beta) * zmp - P(beta) * xmp - zmp * zmp;
        numb kz3 = -P(alpha) * zmp - P(beta) * xmp - P(beta) * ymp - xmp * xmp;

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = -P(alpha) * xmp - P(beta) * ymp - P(beta) * zmp - ymp * ymp;
        numb ky4 = -P(alpha) * ymp - P(beta) * zmp - P(beta) * xmp - zmp * zmp;
        numb kz4 = -P(alpha) * zmp - P(beta) * xmp - P(beta) * ymp - xmp * xmp;

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * P(stepsize) - P(symmetry);
        numb h2 = 0.5 * P(stepsize) + P(symmetry);

        numb xmp = V(x) + h1 * (-P(alpha) * V(x) - P(beta) * V(y) - P(beta) * V(z) - V(y) * V(y));
        numb ymp = V(y) + h1 * (-P(alpha) * V(y) - P(beta) * V(z) - P(beta) * xmp - V(z) * V(z));
        numb zmp = V(z) + h1 * (-P(alpha) * V(z) - P(beta) * xmp - P(beta) * ymp - xmp * xmp);

        Vnext(z) = (zmp - h2 * (P(beta) * (xmp + ymp) + xmp * xmp)) / (1 + h2 * P(alpha));
        Vnext(y) = (ymp - h2 * (P(beta) * (Vnext(z) + xmp) + Vnext(z) * Vnext(z))) / (1 + h2 * P(alpha));
        Vnext(x) = (xmp - h2 * (P(beta) * (Vnext(y) + Vnext(z)) + Vnext(y) * Vnext(y))) / (1 + h2 * P(alpha));
    }

}