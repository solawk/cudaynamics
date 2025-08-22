#include "main.h"
#include "hindmarsh_rose.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, d, r, s, e, Iext, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
    enum maps { LLE };
}

__global__ void kernelProgram_hindmarsh_rose(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_hindmarsh_rose);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_hindmarsh_rose(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_hindmarsh_rose, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_hindmarsh_rose(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + P(Iext));
        Vnext(y) = V(y) + P(stepsize) * (P(c) - P(d) * V(x) * V(x) - V(y));
        Vnext(z) = V(z) + P(stepsize) * (P(r) * (P(s) * (V(x) + P(e)) - V(z)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + P(stepsize) * 0.5 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + P(Iext));
        numb ymp = V(y) + P(stepsize) * 0.5 * (P(c) - P(d) * V(x) * V(x) - V(y));
        numb zmp = V(z) + P(stepsize) * 0.5 * (P(r) * (P(s) * (V(x) + P(e)) - V(z)));

        Vnext(x) = V(x) + P(stepsize) * (ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext));
        Vnext(y) = V(y) + P(stepsize) * (P(c) - P(d) * xmp * xmp - ymp);
        Vnext(z) = V(z) + P(stepsize) * (P(r) * (P(s) * (xmp + P(e)) - zmp));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + P(Iext);
        numb ky1 = P(c) - P(d) * V(x) * V(x) - V(y);
        numb kz1 = P(r) * (P(s) * (V(x) + P(e)) - V(z));

        numb xmp = V(x) + 0.5 * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5 * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5 * P(stepsize) * kz1;

        numb kx2 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext);
        numb ky2 = P(c) - P(d) * xmp * xmp - ymp;
        numb kz2 = P(r) * (P(s) * (xmp + P(e)) - zmp);

        xmp = V(x) + 0.5 * P(stepsize) * kx2;
        ymp = V(y) + 0.5 * P(stepsize) * ky2;
        zmp = V(z) + 0.5 * P(stepsize) * kz2;

        numb kx3 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext);
        numb ky3 = P(c) - P(d) * xmp * xmp - ymp;
        numb kz3 = P(r) * (P(s) * (xmp + P(e)) - zmp);

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext);
        numb ky4 = P(c) - P(d) * xmp * xmp - ymp;
        numb kz4 = P(r) * (P(s) * (xmp + P(e)) - zmp);

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4) / 6.0;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4) / 6.0;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4) / 6.0;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * P(stepsize) - P(symmetry);
        numb h2 = 0.5 * P(stepsize) + P(symmetry);

        numb xmp = V(x) + h1 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + P(Iext));
        numb ymp = V(y) + h1 * (P(c) - P(d) * xmp * xmp - V(y));
        numb zmp = V(z) + h1 * (P(r) * (P(s) * (xmp + P(e)) - V(z)));

        Vnext(z) = (zmp + P(r) * P(s) * (xmp + P(e)) * h2) / (1 + P(r) * h2);
        Vnext(y) = (ymp + (P(c) - P(d) * xmp * xmp) * h2) / (1 + h2);

        Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - Vnext(z) + P(Iext));
        Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * Vnext(x) * Vnext(x) * Vnext(x) + P(b) * Vnext(x) * Vnext(x) - Vnext(z) + P(Iext));
    }
}
