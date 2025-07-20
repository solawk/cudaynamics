#include "main.h"
#include "sprott.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
    enum maps { LLE };
}

__global__ void kernelProgram_sprott(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_sprott);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_sprott(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_sprott, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_sprott(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        numb dx = V(y) + P(a)*V(x)*V(y) + V(x)*V(z);
        numb dy = P(c) - P(b)*V(x)*V(x) + V(y)*V(z);
        numb dz = V(x) - V(x)*V(x) - V(y)*V(y);

        Vnext(x) = V(x) + h * dx;
        Vnext(y) = V(y) + h * dy;
        Vnext(z) = V(z) + h * dz;
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb dx = V(y) + P(a)*V(x)*V(y) + V(x)*V(z);
        numb dy = P(c) - P(b)*V(x)*V(x) + V(y)*V(z);
        numb dz = V(x) - V(x)*V(x) - V(y)*V(y);

        numb xmp = V(x) + h * 0.5 * dx;
        numb ymp = V(y) + h * 0.5 * dy;
        numb zmp = V(z) + h * 0.5 * dz;

        numb dx2 = ymp + P(a)*xmp*ymp + xmp*zmp;
        numb dy2 = P(c) - P(b)*xmp*xmp + ymp*zmp;
        numb dz2 = xmp - xmp*xmp - ymp*ymp;

        Vnext(x) = V(x) + h * dx2;
        Vnext(y) = V(y) + h * dy2;
        Vnext(z) = V(z) + h * dz2;
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {


        numb dx1 = V(y) + P(a) * V(x) * V(y) + V(x) * V(z);
        numb dy1 = P(c) - P(b) * V(x) * V(x) + V(y) * V(z);
        numb dz1 = V(x) - V(x) * V(x) - V(y) * V(y);

        numb xmp = V(x) + 0.5 * h * dx1;
        numb ymp = V(y) + 0.5 * h * dy1;
        numb zmp = V(z) + 0.5 * h * dz1;

        numb dx2 = ymp + P(a) * xmp * ymp + xmp * zmp;
        numb dy2 = P(c) - P(b) * xmp * xmp + ymp * zmp;
        numb dz2 = xmp - xmp * xmp - ymp * ymp;

        xmp = V(x) + 0.5 * h * dx2;
        ymp = V(y) + 0.5 * h * dy2;
        zmp = V(z) + 0.5 * h * dz2;

        numb dx3 = ymp + P(a) * xmp * ymp + xmp * zmp;
        numb dy3 = P(c) - P(b) * xmp * xmp + ymp * zmp;
        numb dz3 = xmp - xmp * xmp - ymp * ymp;

        xmp = V(x) + h * dx3;
        ymp = V(y) + h * dy3;
        zmp = V(z) + h * dz3;

        numb dx4 = ymp + P(a) * xmp * ymp + xmp * zmp;
        numb dy4 = P(c) - P(b) * xmp * xmp + ymp * zmp;
        numb dz4 = xmp - xmp * xmp - ymp * ymp;

        Vnext(x) = V(x) + h * (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6;
        Vnext(y) = V(y) + h * (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6;
        Vnext(z) = V(z) + h * (dz1 + 2 * dz2 + 2 * dz3 + dz4) / 6;

    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb s = P(symmetry);
        numb h1 = 0.5 * h - s;
        numb h2 = 0.5 * h + s;

        numb dx1 = V(y) + P(a) * V(x) * V(y) + V(x) * V(z);
        numb xmp1 = V(x) + h1 * dx1;
        numb dy1 = P(c) - P(b) * xmp1 * xmp1 + V(y) * V(z);
        numb ymp1 = V(y) + h1 * dy1;
        numb dz1 = xmp1 - xmp1 * xmp1 - ymp1 * ymp1;
        numb zmp1 = V(z) + h1 * dz1;

        numb xmp2 = xmp1;
        numb ymp2 = ymp1;
        numb zmp2 = zmp1;

        zmp1 = zmp2+h2*(xmp1-xmp1*xmp1-ymp1*ymp1);
        ymp1 = (ymp2+h2*P(c)-h2*P(b)*xmp1*xmp1)/(1-h2*zmp1);
        xmp1 = (xmp2+h2*ymp1)/(1-h2*P(a)*ymp1-h2*zmp1);

        Vnext(x) = xmp1;
        Vnext(y) = ymp1;
        Vnext(z) = zmp1;
    }
}