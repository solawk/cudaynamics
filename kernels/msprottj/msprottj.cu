#include "main.h"
#include "msprottj.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, method };
    enum methods { ExplicitEuler, ExplicitRK4 };
    enum maps { LLE };
}

__global__ void kernelProgram_msprottj(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_msprottj);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_msprottj(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_msprottj);
    }
}

__device__ numb msprottj_F(numb y)
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

__device__ void finiteDifferenceScheme_msprottj(numb* currentV, numb* nextV, numb* parameters, numb h)
{  
    ifMETHOD(P(method), ExplicitEuler)
    {
        numb dx = P(a) * V(z);
        numb dy = P(b) * V(y) + V(z);
        numb dz = -V(x) + V(y) + P(c) * msprottj_F(V(y));

        Vnext(x) = V(x) + h * dx;
        Vnext(y) = V(y) + h * dy;
        Vnext(z) = V(z) + h * dz;
    }

    ifMETHOD(P(method), ExplicitRK4)
    {
        numb dx1 = P(a) * V(z);
        numb dy1 = P(b) * V(y) + V(z);
        numb dz1 = -V(x) + V(y) + P(c) * msprottj_F(V(y));

        numb xt = V(x) + 0.5 * h * dx1;
        numb yt = V(y) + 0.5 * h * dy1;
        numb zt = V(z) + 0.5 * h * dz1;

        numb dx2 = P(a) * zt;
        numb dy2 = P(b) * yt + zt;
        numb dz2 = -xt + yt + P(c) * msprottj_F(yt);

        xt = V(x) + 0.5 * h * dx2;
        yt = V(y) + 0.5 * h * dy2;
        zt = V(z) + 0.5 * h * dz2;

        numb dx3 = P(a) * zt;
        numb dy3 = P(b) * yt + zt;
        numb dz3 = -xt + yt + P(c) * msprottj_F(yt);

        xt = V(x) + h * dx3;
        yt = V(y) + h * dy3;
        zt = V(z) + h * dz3;

        numb dx4 = P(a) * zt;
        numb dy4 = P(b) * yt + zt;
        numb dz4 = -xt + yt + P(c) * msprottj_F(yt);

        Vnext(x) = V(x) + h * (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6;
        Vnext(y) = V(y) + h * (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6;
        Vnext(z) = V(z) + h * (dz1 + 2 * dz2 + 2 * dz3 + dz4) / 6;
    }
}