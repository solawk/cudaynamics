#include "main.h"
#include "langford.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, d, e, f, g, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_langford(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_langford);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_langford(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_langford, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_langford, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4), P(stepsize));
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_langford, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}


__device__ __forceinline__ void finiteDifferenceScheme_langford(numb* currentV, numb* nextV, numb* parameters, Computation* data)
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