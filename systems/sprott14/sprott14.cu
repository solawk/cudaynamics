#include "main.h"
#include "sprott14.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_sprott14(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES;

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_sprott14);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_sprott14(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_sprott14, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_sprott14, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4), P(stepsize));
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_sprott14, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__ void finiteDifferenceScheme_sprott14(numb* currentV, numb* nextV, numb* parameters, Computation* data)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (V(y) + P(a) * V(x) * V(y) + V(x) * V(z));
        Vnext(y) = V(y) + P(stepsize) * (P(c) - P(b) * V(x) * V(x) + V(y) * V(z));
        Vnext(z) = V(z) + P(stepsize) * (V(x) - V(x) * V(x) - V(y) * V(y));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + P(stepsize) * 0.5 * (V(y) + P(a) * V(x) * V(y) + V(x) * V(z));
        numb ymp = V(y) + P(stepsize) * 0.5 * (P(c) - P(b) * V(x) * V(x) + V(y) * V(z));
        numb zmp = V(z) + P(stepsize) * 0.5 * (V(x) - V(x) * V(x) - V(y) * V(y));

        Vnext(x) = V(x) + P(stepsize) * (ymp + P(a) * xmp * ymp + xmp * zmp);
        Vnext(y) = V(y) + P(stepsize) * (P(c) - P(b) * xmp * xmp + ymp * zmp);
        Vnext(z) = V(z) + P(stepsize) * (xmp - xmp * xmp - ymp * ymp);
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = V(y) + P(a) * V(x) * V(y) + V(x) * V(z);
        numb ky1 = P(c) - P(b) * V(x) * V(x) + V(y) * V(z);
        numb kz1 = V(x) - V(x) * V(x) - V(y) * V(y);

        numb xmp = V(x) + 0.5 * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5 * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5 * P(stepsize) * kz1;

        numb kx2 = ymp + P(a) * xmp * ymp + xmp * zmp;
        numb ky2 = P(c) - P(b) * xmp * xmp + ymp * zmp;
        numb kz2 = xmp - xmp * xmp - ymp * ymp;

        xmp = V(x) + 0.5 * P(stepsize) * kx2;
        ymp = V(y) + 0.5 * P(stepsize) * ky2;
        zmp = V(z) + 0.5 * P(stepsize) * kz2;

        numb kx3 = ymp + P(a) * xmp * ymp + xmp * zmp;
        numb ky3 = P(c) - P(b) * xmp * xmp + ymp * zmp;
        numb kz3 = xmp - xmp * xmp - ymp * ymp;

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = ymp + P(a) * xmp * ymp + xmp * zmp;
        numb ky4 = P(c) - P(b) * xmp * xmp + ymp * zmp;
        numb kz4 = xmp - xmp * xmp - ymp * ymp;

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * P(stepsize) - P(symmetry);
        numb h2 = 0.5 * P(stepsize) + P(symmetry);

        numb xmp = V(x) + h1 * (V(y) + P(a) * V(x) * V(y) + V(x) * V(z));
        numb ymp = V(y) + h1 * (P(c) - P(b) * xmp * xmp + V(y) * V(z));
        numb zmp = V(z) + h1 * (xmp - xmp * xmp - ymp * ymp);

        Vnext(z) = zmp + h2 * (xmp - xmp * xmp - ymp * ymp);
        Vnext(y) = (ymp + h2 * P(c) - h2 * P(b) * xmp * xmp) / (1 - h2 * Vnext(z));
        Vnext(x) = (xmp + h2 * Vnext(y)) / (1 - h2 * P(a) * Vnext(y) - h2 * Vnext(z));
    }
}