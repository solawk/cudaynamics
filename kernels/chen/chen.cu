#include "main.h"
#include "chen.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { alpha, beta, gamma, delta, symmetry, method };
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
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_chen);
    }
}

__device__ void finiteDifferenceScheme_chen(numb* currentV, numb* nextV, numb* parameters, numb h)
{

    ifMETHOD(P(method), ExplicitEuler)
    {
        numb dx = P(alpha)*V(x) - V(y)*V(z);
        numb dy = P(beta)*V(y) + V(x)*V(z);
        numb dz = P(delta)*V(z) + V(x)*V(y)/P(gamma);

        Vnext(x) = V(x) + h * dx;
        Vnext(y) = V(y) + h * dy;
        Vnext(z) = V(z) + h * dz;
    }


    ifMETHOD(P(method), ExplicitMidpoint)
    {
        h *= 0.5;
        numb x1 = V(x) + h * (P(alpha) * V(x) - V(y) * V(z));
        numb y1 = V(y) + h * (P(beta) * V(y) + V(x) * V(z));
        numb z1 = V(z) + h * (P(delta) * V(z) + V(x) * V(y) / P(gamma));

        h *= 2;
        Vnext(x) = V(x) + h * (P(alpha) * x1 - y1 * z1);
        Vnext(y) = V(y) + h * (P(beta) * y1 + x1 * z1);
        Vnext(z) = V(z) + h * (P(delta) * z1 + x1 * y1 / P(gamma));
    }


    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(alpha) * V(x) - V(y) * V(z);
        numb ky1 = P(beta) * V(y) + V(x) * V(z);
        numb kz1 = P(delta) * V(z) + V(x) * V(y) / P(gamma);

        numb dx = V(x) + 0.5 * h * kx1;
        numb dy = V(y) + 0.5 * h * ky1;
        numb dz = V(z) + 0.5 * h * kz1;

        numb kx2 = P(alpha) * dx - dy * dz;
        numb ky2 = P(beta) * dy + dx * dz;
        numb kz2 = P(delta) * dz + dx * dy / P(gamma);

        dx = V(x) + 0.5 * h * kx2;
        dy = V(y) + 0.5 * h * ky2;
        dz = V(z) + 0.5 * h * kz2;

        numb kx3 = P(alpha) * dx - dy * dz;
        numb ky3 = P(beta) * dy + dx * dz;
        numb kz3 = P(delta) * dz + dx * dy / P(gamma);

        dx = V(x) + h * kx3;
        dy = V(y) + h * ky3;
        dz = V(z) + h * kz3;

        numb kx4 = P(alpha) * dx - dy * dz;
        numb ky4 = P(beta) * dy + dx * dz;
        numb kz4 = P(delta) * dz + dx * dy / P(gamma);

        Vnext(x) = V(x) + h * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + h * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + h * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }


    ifMETHOD(P(method), VariableSymmetryCD)
    {
        

    }

}