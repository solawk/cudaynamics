#include "main.h"
#include "three_scroll.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, d, e, f, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE };
}

__global__ void kernelProgram_three_scroll(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_three_scroll);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_three_scroll(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_three_scroll);
    }
}

__device__ void finiteDifferenceScheme_three_scroll(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        numb dx = P(a) * (V(y) - V(x)) + P(d) * V(x) * V(z);
        numb dy = V(x) * (P(b) - V(z)) + P(f) * V(y);
        numb dz = P(c) * V(z) + V(x) * (V(y) - P(e) * V(x));

        Vnext(x) = V(x) + h * dx;
        Vnext(y) = V(y) + h * dy;
        Vnext(z) = V(z) + h * dz;
    }


    ifMETHOD(P(method), ExplicitMidpoint)
    {
        h *= 0.5;
        numb x1 = V(x) + h * (P(a) * (V(y) - V(x)) + P(d) * V(x) * V(z));
        numb y1 = V(y) + h * (V(x) * (P(b) - V(z)) + P(f) * V(y));
        numb z1 = V(z) + h * (P(c) * V(z) + V(x) * (V(y) - P(e) * V(x)));

        h *= 2;
        Vnext(x) = V(x) + h * (P(a) * (y1 - x1) + P(d) * x1 * z1);
        Vnext(y) = V(y) + h * (x1 * (P(b) - z1) + P(f) * y1);
        Vnext(z) = V(z) + h * (P(c) * z1 + x1 * (y1 - P(e) * x1));
    }


    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(a) * (V(y) - V(x)) + P(d) * V(x) * V(z);
        numb ky1 = V(x) * (P(b) - V(z)) + P(f) * V(y);
        numb kz1 = P(c) * V(z) + V(x) * (V(y) - P(e) * V(x));

        numb dx = V(x) + 0.5 * h * kx1;
        numb dy = V(y) + 0.5 * h * ky1;
        numb dz = V(z) + 0.5 * h * kz1;

        numb kx2 = P(a) * (dy - dx) + P(d) * dx * dz;
        numb ky2 = dx * (P(b) - dz) + P(f) * dy;
        numb kz2 = P(c) * dz + dx * (dy - P(e) * dx);

        dx = V(x) + 0.5 * h * kx2;
        dy = V(y) + 0.5 * h * ky2;
        dz = V(z) + 0.5 * h * kz2;

        numb kx3 = P(a) * (dy - dx) + P(d) * dx * dz;
        numb ky3 = dx * (P(b) - dz) + P(f) * dy;
        numb kz3 = P(c) * dz + dx * (dy - P(e) * dx);

        dx = V(x) + h * kx3;
        dy = V(y) + h * ky3;
        dz = V(z) + h * kz3;

        numb kx4 = P(a) * (dy - dx) + P(d) * dx * dz;
        numb ky4 = dx * (P(b) - dz) + P(f) * dy;
        numb kz4 = P(c) * dz + dx * (dy - P(e) * dx);

        Vnext(x) = V(x) + h * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + h * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + h * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }


    ifMETHOD(P(method), VariableSymmetryCD)
    {
        
    }
}