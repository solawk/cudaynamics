#include "main.h"
#include "langford.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, d, e, f, g, symmetry, method };
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
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_langford);
    }
}

__device__ void finiteDifferenceScheme_langford(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        numb dx = (V(z) - P(b)) * V(x) - P(d) * V(y);
        numb dy = P(d) * V(x) + (V(z) - P(b)) * V(y);
        numb dz = P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (V(x) * V(x) + V(y) * V(y)) * (1 + P(e) * V(z)) + P(f) * V(z);

        Vnext(x) = V(x) + h * dx;
        Vnext(y) = V(y) + h * dy;
        Vnext(z) = V(z) + h * dz;
    }


    ifMETHOD(P(method), ExplicitMidpoint)
    {
        h *= 0.5;
        numb x1 = V(x) + h * ((V(z) - P(b)) * V(x) - P(d) * V(y));
        numb y1 = V(y) + h * (P(d) * V(x) + (V(z) - P(b)) * V(y));
        numb z1 = V(z) + h * (P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (V(x) * V(x) + V(y) * V(y)) * (1 + P(e) * V(z)) + P(f) * V(z));

        h *= 2;
        Vnext(x) = V(x) + h * ((z1 - P(b)) * x1 - P(d) * y1);
        Vnext(y) = V(y) + h * (P(d) * x1 + (z1 - P(b)) * y1);
        Vnext(z) = V(z) + h * (P(c) + P(a) * z1 - z1 * z1 * z1 / P(g) - (x1 * x1 + y1 * y1) * (1 + P(e) * z1) + P(f) * z1);
    }


    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = (V(z) - P(b)) * V(x) - P(d) * V(y);
        numb ky1 = P(d) * V(x) + (V(z) - P(b)) * V(y);
        numb kz1 = P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (V(x) * V(x) + V(y) * V(y)) * (1 + P(e) * V(z)) + P(f) * V(z);

        numb dx = V(x) + 0.5 * h * kx1;
        numb dy = V(y) + 0.5 * h * ky1;
        numb dz = V(z) + 0.5 * h * kz1;

        numb kx2 = (dz - P(b)) * dx - P(d) * dy;
        numb ky2 = P(d) * dx + (dz - P(b)) * dy;
        numb kz2 = P(c) + P(a) * dz - dz * dz * dz / P(g) - (dx * dx + dy * dy) * (1 + P(e) * dz) + P(f) * dz;

        dx = V(x) + 0.5 * h * kx2;
        dy = V(y) + 0.5 * h * ky2;
        dz = V(z) + 0.5 * h * kz2;

        numb kx3 = (dz - P(b)) * dx - P(d) * dy;
        numb ky3 = P(d) * dx + (dz - P(b)) * dy;
        numb kz3 = P(c) + P(a) * dz - dz * dz * dz / P(g) - (dx * dx + dy * dy) * (1 + P(e) * dz) + P(f) * dz;

        dx = V(x) + h * kx3;
        dy = V(y) + h * ky3;
        dz = V(z) + h * kz3;

        numb kx4 = (dz - P(b)) * dx - P(d) * dy;
        numb ky4 = P(d) * dx + (dz - P(b)) * dy;
        numb kz4 = P(c) + P(a) * dz - dz * dz * dz / P(g) - (dx * dx + dy * dy) * (1 + P(e) * dz) + P(f) * dz;

        Vnext(x) = V(x) + h * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + h * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + h * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }


    ifMETHOD(P(method), VariableSymmetryCD)
    {
        float s = 0;
        float h1 = h * 0.5 - s;
        float h2 = h * 0.5 + s;

        numb x1 = V(x) + h1 * ((V(z) - P(b)) * V(x) - P(d) * V(y));
        numb y1 = V(y) + h1 * (P(d) * x1 + (V(z) - P(b)) * V(y));
        numb z1 = V(z) + h1 * (P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (x1 * x1 + y1 * y1) * (1 + P(e) * V(z)) + P(f) * V(z));

        numb z2 = z1 + h2 * (P(c) + P(a) * z1 - z1 * z1 * z1 / P(g) - (x1 * x1 + y1 * y1) * (1 + P(e) * z1) + P(f) * z1);
        z2 = z1 + h2 * (P(c) + P(a) * z2 - z2 * z2 * z2 / P(g) - (x1 * x1 + y1 * y1) * (1 + P(e) * z2) + P(f) * z2);
        numb y2 = (y1 + h2 * P(d) * x1) / (1 - h2 * (z2 - P(b)));
        numb x2 = (x1 - h2 * P(d) * y2) / (1 - h2 * (z2 - P(b)));

        Vnext(z) = z2;
        Vnext(y) = y2;
        Vnext(x) = x2;
    }

}