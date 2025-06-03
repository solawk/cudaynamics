#include "main.h"
#include "halvorsen.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { alpha, beta, symmetry, method };
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
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_halvorsen);
    }
}

__device__ void finiteDifferenceScheme_halvorsen(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        numb dx = -P(alpha) * V(x) - P(beta) * V(y) - P(beta) * V(z) - V(y) * V(y);
        numb dy = -P(alpha) * V(y) - P(beta) * V(z) - P(beta) * V(x) - V(z) * V(z);
        numb dz = -P(alpha) * V(z) - P(beta) * V(x) - P(beta) * V(y) - V(x) * V(x);

        Vnext(x) = V(x) + h * dx;
        Vnext(y) = V(y) + h * dy;
        Vnext(z) = V(z) + h * dz;
    }


    ifMETHOD(P(method), ExplicitMidpoint)
    {
        h *= 0.5;
        numb x1 = V(x) + h * (-P(alpha) * V(x) - P(beta) * V(y) - P(beta) * V(z) - V(y) * V(y));
        numb y1 = V(y) + h * (-P(alpha) * V(y) - P(beta) * V(z) - P(beta) * V(x) - V(z) * V(z));
        numb z1 = V(z) + h * (-P(alpha) * V(z) - P(beta) * V(x) - P(beta) * V(y) - V(x) * V(x));

        h *= 2;
        Vnext(x) = V(x) + h * (-P(alpha) * x1 - P(beta) * y1 - P(beta) * z1 - y1 * y1);
        Vnext(y) = V(y) + h * (-P(alpha) * y1 - P(beta) * z1 - P(beta) * x1 - z1 * z1);
        Vnext(z) = V(z) + h * (-P(alpha) * z1 - P(beta) * x1 - P(beta) * y1 - x1 * x1);
    }


    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = -P(alpha) * V(x) - P(beta) * V(y) - P(beta) * V(z) - V(y) * V(y);
        numb ky1 = -P(alpha) * V(y) - P(beta) * V(z) - P(beta) * V(x) - V(z) * V(z);
        numb kz1 = -P(alpha) * V(z) - P(beta) * V(x) - P(beta) * V(y) - V(x) * V(x);

        numb dx = V(x) + 0.5 * h * kx1;
        numb dy = V(y) + 0.5 * h * ky1;
        numb dz = V(z) + 0.5 * h * kz1;

        numb kx2 = -P(alpha) * dx - P(beta) * dy - P(beta) * dz - dy * dy;
        numb ky2 = -P(alpha) * dy - P(beta) * dz - P(beta) * dx - dz * dz;
        numb kz2 = -P(alpha) * dz - P(beta) * dx - P(beta) * dy - dx * dx;

        dx = V(x) + 0.5 * h * kx2;
        dy = V(y) + 0.5 * h * ky2;
        dz = V(z) + 0.5 * h * kz2;

        numb kx3 = -P(alpha) * dx - P(beta) * dy - P(beta) * dz - dy * dy;
        numb ky3 = -P(alpha) * dy - P(beta) * dz - P(beta) * dx - dz * dz;
        numb kz3 = -P(alpha) * dz - P(beta) * dx - P(beta) * dy - dx * dx;

        dx = V(x) + h * kx3;
        dy = V(y) + h * ky3;
        dz = V(z) + h * kz3;

        numb kx4 = -P(alpha) * dx - P(beta) * dy - P(beta) * dz - dy * dy;
        numb ky4 = -P(alpha) * dy - P(beta) * dz - P(beta) * dx - dz * dz;
        numb kz4 = -P(alpha) * dz - P(beta) * dx - P(beta) * dy - dx * dx;

        Vnext(x) = V(x) + h * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + h * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + h * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        float s = 0;
        float h1 = h * 0.5 - s;
        float h2 = h * 0.5 + s;

        numb x1 = V(x) + h1 * (-P(alpha) * V(x) - P(beta) * V(y) - P(beta) * V(z) - V(y) * V(y));
        numb y1 = V(y) + h1 * (-P(alpha) * V(y) - P(beta) * V(z) - P(beta) * x1 - V(z) * V(z));
        numb z1 = V(z) + h1 * (-P(alpha) * V(z) - P(beta) * x1 - P(beta) * y1 - x1 * x1);

        numb z2 = (z1 - h2 * (P(beta) * (x1 + y1) + x1 * x1)) / (1 + h2 * P(alpha));
        numb y2 = (y1 - h2 * (P(beta) * (z2 + x1) + z2 * z2)) / (1 + h2 * P(alpha));
        numb x2 = (x1 - h2 * (P(beta) * (y2 + z2) + y2 * y2)) / (1 + h2 * P(alpha));

        Vnext(z) = z2;
        Vnext(y) = y2;
        Vnext(x) = x2;
    }

}