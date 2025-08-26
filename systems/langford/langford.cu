#include "main.h"
#include "langford.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, d, e, f, g, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE };
}

__global__ void kernelProgram_langford(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_langford);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_langford(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]));
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_langford, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_langford(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * ((V(z) - P(b)) * V(x) - P(d) * V(y));
        Vnext(y) = V(y) + P(stepsize) * (P(d) * V(x) + (V(z) - P(b)) * V(y));
        Vnext(z) = V(z) + P(stepsize) * (P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (V(x) * V(x) + V(y) * V(y)) * (1 + P(e) * V(z)) + P(f) * V(z));
    }


    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + 0.5 * P(stepsize) * ((V(z) - P(b)) * V(x) - P(d) * V(y));
        numb ymp = V(y) + 0.5 * P(stepsize) * (P(d) * V(x) + (V(z) - P(b)) * V(y));
        numb zmp = V(z) + 0.5 * P(stepsize) * (P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (V(x) * V(x) + V(y) * V(y)) * (1 + P(e) * V(z)) + P(f) * V(z));

        Vnext(x) = V(x) + P(stepsize) * ((zmp - P(b)) * xmp - P(d) * ymp);
        Vnext(y) = V(y) + P(stepsize) * (P(d) * xmp + (zmp - P(b)) * ymp);
        Vnext(z) = V(z) + P(stepsize) * (P(c) + P(a) * zmp - zmp * zmp * zmp / P(g) - (xmp * xmp + ymp * ymp) * (1 + P(e) * zmp) + P(f) * zmp);
    }


    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = (V(z) - P(b)) * V(x) - P(d) * V(y);
        numb ky1 = P(d) * V(x) + (V(z) - P(b)) * V(y);
        numb kz1 = P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (V(x) * V(x) + V(y) * V(y)) * (1 + P(e) * V(z)) + P(f) * V(z);

        numb xmp = V(x) + 0.5 * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5 * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5 * P(stepsize) * kz1;

        numb kx2 = (zmp - P(b)) * xmp - P(d) * ymp;
        numb ky2 = P(d) * xmp + (zmp - P(b)) * ymp;
        numb kz2 = P(c) + P(a) * zmp - zmp * zmp * zmp / P(g) - (xmp * xmp + ymp * ymp) * (1 + P(e) * zmp) + P(f) * zmp;

        xmp = V(x) + 0.5 * P(stepsize) * kx2;
        ymp = V(y) + 0.5 * P(stepsize) * ky2;
        zmp = V(z) + 0.5 * P(stepsize) * kz2;

        numb kx3 = (zmp - P(b)) * xmp - P(d) * ymp;
        numb ky3 = P(d) * xmp + (zmp - P(b)) * ymp;
        numb kz3 = P(c) + P(a) * zmp - zmp * zmp * zmp / P(g) - (xmp * xmp + ymp * ymp) * (1 + P(e) * zmp) + P(f) * zmp;

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = (zmp - P(b)) * xmp - P(d) * ymp;
        numb ky4 = P(d) * xmp + (zmp - P(b)) * ymp;
        numb kz4 = P(c) + P(a) * zmp - zmp * zmp * zmp / P(g) - (xmp * xmp + ymp * ymp) * (1 + P(e) * zmp) + P(f) * zmp;

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }


    ifMETHOD(P(method), VariableSymmetryCD)
    {
        float h1 = 0.5 * P(stepsize) - P(symmetry);
        float h2 = 0.5 * P(stepsize) + P(symmetry);

        numb xmp = V(x) + h1 * ((V(z) - P(b)) * V(x) - P(d) * V(y));
        numb ymp = V(y) + h1 * (P(d) * xmp + (V(z) - P(b)) * V(y));
        numb zmp = V(z) + h1 * (P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (xmp * xmp + ymp * ymp) * (1 + P(e) * V(z)) + P(f) * V(z));

        Vnext(z) = zmp + h2 * (P(c) + P(a) * zmp - zmp * zmp * zmp / P(g) - (xmp * xmp + ymp * ymp) * (1 + P(e) * zmp) + P(f) * zmp);
        Vnext(z) = zmp + h2 * (P(c) + P(a) * Vnext(z) - Vnext(z) * Vnext(z) * Vnext(z) / P(g) - (xmp * xmp + ymp * ymp) * (1 + P(e) * Vnext(z)) + P(f) * Vnext(z));
        Vnext(y) = (ymp + h2 * P(d) * xmp) / (1 - h2 * (Vnext(z) - P(b)));
        Vnext(x) = (xmp - h2 * P(d) * Vnext(y)) / (1 - h2 * (Vnext(z) - P(b)));
    }
}