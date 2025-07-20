#include "main.h"
#include "rossler.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
    enum maps { LLE };
}

__global__ void kernelProgram_rossler(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_rossler);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_rossler(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_rossler, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_rossler(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        numb dx = -(V(y) + V(z));
        numb dy = V(x) + P(a)*V(y);
        numb dz = P(b) + V(z)*(V(x) - P(c));

        Vnext(x) = V(x) + h * dx;
        Vnext(y) = V(y) + h * dy;
        Vnext(z) = V(z) + h * dz;
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb dx = -(V(y) + V(z));
        numb dy = V(x) + P(a)*V(y);
        numb dz = P(b) + V(z)*(V(x)-P(c));

        numb xmp = V(x) + h * 0.5 * dx;
        numb ymp = V(y) + h * 0.5 * dy;
        numb zmp = V(z) + h * 0.5 * dz;

        numb dx2 = -(ymp + zmp);
        numb dy2 = xmp + P(a)*ymp;
        numb dz2 = P(b) + zmp*(xmp - P(c));

        Vnext(x) = V(x) + h * dx2;
        Vnext(y) = V(y) + h * dy2;
        Vnext(z) = V(z) + h * dz2;
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {


        numb dx1 = -(V(y) + V(z));
        numb dy1 = V(x) + P(a) * V(y);
        numb dz1 = P(b) + V(z) * (V(x) - P(c));

        numb xmp = V(x) + 0.5 * h * dx1;
        numb ymp = V(y) + 0.5 * h * dy1;
        numb zmp = V(z) + 0.5 * h * dz1;

        numb dx2 = -(ymp + zmp);
        numb dy2 = xmp + P(a) * ymp;
        numb dz2 = P(b) + zmp * (xmp - P(c));

        xmp = V(x) + 0.5 * h * dx2;
        ymp = V(y) + 0.5 * h * dy2;
        zmp = V(z) + 0.5 * h * dz2;

        numb dx3 = -(ymp + zmp);
        numb dy3 = xmp + P(a) * ymp;
        numb dz3 = P(b) + zmp * (xmp - P(c));

        xmp = V(x) + h * dx3;
        ymp = V(y) + h * dy3;
        zmp = V(z) + h * dz3;

        numb dx4 = -(ymp + zmp);
        numb dy4 = xmp + P(a) * ymp;
        numb dz4 = P(b) + zmp * (xmp - P(c));

        Vnext(x) = V(x) + h * (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6;
        Vnext(y) = V(y) + h * (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6;
        Vnext(z) = V(z) + h * (dz1 + 2 * dz2 + 2 * dz3 + dz4) / 6;

    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb s = P(symmetry);
        numb h1 = 0.5 * h - s;
        numb h2 = 0.5 * h + s;

        numb dx1 = -(V(y) + V(z));
        numb xmp1 = V(x) + h1 * dx1;
        numb dy1 = xmp1 + P(a) * V(y);
        numb ymp1 = V(y) + h1 * dy1;
        numb dz1 = P(b) + V(z) * (xmp1 - P(c));
        numb zmp1 = V(z) + h1 * dz1;

        numb zmp2 = (zmp1+h2*P(b))/(1-h2*xmp1+h2*P(c));
        numb ymp2 = (ymp1+xmp1*h2)/(1-P(a)*h2);
        numb xmp2 = xmp1-h2*ymp2-h2*zmp2;

        Vnext(x) = xmp2;
        Vnext(y) = ymp2;
        Vnext(z) = zmp2;
    }
}