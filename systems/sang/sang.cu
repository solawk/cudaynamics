#include "main.h"
#include "sang.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, mu, omega, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE };
}

__global__ void kernelProgram_sang(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_sang);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_sang(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]));
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_sang, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_sang(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (-V(y));
        Vnext(y) = V(y) + P(stepsize) * (V(x) + P(c) * V(y) + P(a) * V(z));
        Vnext(z) = V(z) + P(stepsize) * (-P(mu) * V(z) + P(b) * cosf(P(omega) * V(y)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + P(stepsize) * 0.5 * (-V(y));
        numb ymp = V(y) + P(stepsize) * 0.5 * (V(x) + P(c) * V(y) + P(a) * V(z));
        numb zmp = V(z) + P(stepsize) * 0.5 * (-P(mu) * V(z) + P(b) * cosf(P(omega) * V(y)));

        Vnext(x) = V(x) + P(stepsize) * (-ymp);
        Vnext(y) = V(y) + P(stepsize) * (xmp + P(c) * ymp + P(a) * zmp);
        Vnext(z) = V(z) + P(stepsize) * (-P(mu) * zmp + P(b) * cosf(P(omega) * ymp));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = -V(y);
        numb ky1 = V(x) + P(c) * V(y) + P(a) * V(z);
        numb kz1 = -P(mu) * V(z) + P(b) * cosf(P(omega) * V(y));

        numb xmp = V(x) + 0.5 * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5 * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5 * P(stepsize) * kz1;

        numb kx2 = -ymp;
        numb ky2 = xmp + P(c) * ymp + P(a) * zmp;
        numb kz2 = -P(mu) * zmp + P(b) * cosf(P(omega) * ymp);

        xmp = V(x) + 0.5 * P(stepsize) * kx2;
        ymp = V(y) + 0.5 * P(stepsize) * ky2;
        zmp = V(z) + 0.5 * P(stepsize) * kz2;

        numb kx3 = -ymp;
        numb ky3 = xmp + P(c) * ymp + P(a) * zmp;
        numb kz3 = -P(mu) * zmp + P(b) * cosf(P(omega) * ymp);

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = -ymp;
        numb ky4 = xmp + P(c) * ymp + P(a) * zmp;
        numb kz4 = -P(mu) * zmp + P(b) * cosf(P(omega) * ymp);

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * P(stepsize) - P(symmetry);
        numb h2 = 0.5 * P(stepsize) + P(symmetry);

        numb xmp = V(x) + h1 * (-V(y));
        numb ymp = V(y) + h1 * (xmp + P(c) * V(y) + P(a) * V(z));
        numb zmp = V(z) + h1 * (-P(mu) * V(z) + P(b) * cosf(P(omega) * ymp));

        Vnext(z) = (zmp + P(b) * cosf(P(omega) * ymp) * h2) / (1 + P(mu) * h2);
        Vnext(y) = (ymp + (xmp + P(a) * Vnext(z)) * h2) / (1 - P(c) * h2);
        Vnext(x) = xmp + h2 * (-Vnext(y));
    }
}