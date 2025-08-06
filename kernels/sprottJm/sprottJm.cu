#include "main.h"
#include "sprottJm.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, stepsize, method };
    enum methods { ExplicitEuler, ExplicitRungeKutta4 };
    enum maps { LLE };
}

__global__ void kernelProgram_sprottJm(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_sprottJm);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_sprottJm(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_sprottJm, MO(LLE));
    }
}

__device__ numb sprottJm_F(numb y)
{
    numb b, k;
    numb d = 1.0, m = 1.0, P = 1.23, R = 2.0;
    numb ay = abs(y);
    while (true)
    {
        if (ay < d)
        {
            d /= R; m /= R;
        }
        else if (ay > 2 * d)
        {
            d *= R; m *= R;
        }
        else
            break;
    }
    numb epsilon = 0.01;
    if (d > epsilon)
    {
        if (ay < P * d)
        {
            b = -m * (R - P * R + 1) / (R * (P - 1));
            k = -m / (R * d * (1 - P));
        }
        else
        {
            b = -m * (R - P * R + 1) / (P - R);
            k = -m * (-(R * R) + R + 1) / (R * d * (P - R));
        }

        return k * ay + b;
    }
    else
        return 0.0;
}

__device__ void finiteDifferenceScheme_sprottJm(numb* currentV, numb* nextV, numb* parameters, numb h)
{  
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (P(a) * V(z));
        Vnext(y) = V(y) + P(stepsize) * (P(b) * V(y) + V(z));
        Vnext(z) = V(z) + P(stepsize) * (-V(x) + V(y) + P(c) * sprottJm_F(V(y)));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(a) * V(z);
        numb ky1 = P(b) * V(y) + V(z);
        numb kz1 = -V(x) + V(y) + P(c) * sprottJm_F(V(y));

        numb xmp = V(x) + 0.5 * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5 * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5 * P(stepsize) * kz1;

        numb kx2 = P(a) * zmp;
        numb ky2 = P(b) * ymp + zmp;
        numb kz2 = -xmp + ymp + P(c) * sprottJm_F(ymp);

        xmp = V(x) + 0.5 * P(stepsize) * kx2;
        ymp = V(y) + 0.5 * P(stepsize) * ky2;
        zmp = V(z) + 0.5 * P(stepsize) * kz2;

        numb kx3 = P(a) * zmp;
        numb ky3 = P(b) * ymp + zmp;
        numb kz3 = -xmp + ymp + P(c) * sprottJm_F(ymp);

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = P(a) * zmp;
        numb ky4 = P(b) * ymp + zmp;
        numb kz4 = -xmp + ymp + P(c) * sprottJm_F(ymp);

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }
}