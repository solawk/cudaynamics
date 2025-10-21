#include "main.h"
#include "dadras.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, d, e, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_dadras(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_dadras);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_dadras(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_dadras, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_dadras, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4),
            H_BRANCH(parameters[CUDA_kernel.PARAM_COUNT - 1], variables[CUDA_kernel.VAR_COUNT - 1]));
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_dadras, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__ void finiteDifferenceScheme_dadras(numb* currentV, numb* nextV, numb* parameters, Computation* data)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + H * (V(y) + P(a) * V(x) + P(b) * V(y) * V(z));
        Vnext(y) = V(y) + H * (P(c) * V(y) - V(x) * V(z) + V(z));
        Vnext(z) = V(z) + H * (P(d) * V(x) * V(y) - P(e) * V(z));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * 0.5 * (V(y) - P(a) * V(x) + P(b) * V(y) * V(z));
        numb ymp = V(y) + H * 0.5 * (P(c) * V(y) - V(x) * V(z) + V(z));
        numb zmp = V(z) + H * 0.5 * (P(d) * V(x) * V(y) - P(e) * V(z));

        Vnext(x) = V(x) + H * (ymp - P(a) * xmp + P(b) * ymp * zmp);
        Vnext(y) = V(y) + H * (P(c) * ymp - xmp * zmp + zmp);
        Vnext(z) = V(z) + H * (P(d) * xmp * ymp - P(e) * zmp);
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = V(y) - P(a) * V(x) + P(b) * V(y) * V(z);
        numb ky1 = P(c) * V(y) - V(x) * V(z) + V(z);
        numb kz1 = P(d) * V(x) * V(y) - P(e) * V(z);

        numb xmp = V(x) + 0.5 * H * kx1;
        numb ymp = V(y) + 0.5 * H * ky1;
        numb zmp = V(z) + 0.5 * H * kz1;

        numb kx2 = ymp - P(a) * xmp + P(b) * ymp * zmp;
        numb ky2 = P(c) * ymp - xmp * zmp + zmp;
        numb kz2 = P(d) * xmp * ymp - P(e) * zmp;

        xmp = V(x) + 0.5 * H * kx2;
        ymp = V(y) + 0.5 * H * ky2;
        zmp = V(z) + 0.5 * H * kz2;

        numb kx3 = ymp - P(a) * xmp + P(b) * ymp * zmp;
        numb ky3 = P(c) * ymp - xmp * zmp + zmp;
        numb kz3 = P(d) * xmp * ymp - P(e) * zmp;

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = ymp - P(a) * xmp + P(b) * ymp * zmp;
        numb ky4 = P(c) * ymp - xmp * zmp + zmp;
        numb kz4 = P(d) * xmp * ymp - P(e) * zmp;

        Vnext(x) = V(x) + H * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + H * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + H * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (V(y) - P(a) * V(x) + P(b) * V(y) * V(z));
        numb ymp = V(y) + h1 * (P(c) * V(y) - xmp * V(z) + V(z));
        numb zmp = V(z) + h1 * (P(d) * xmp * ymp - P(e) * V(z));
        
        Vnext(z) = (zmp + P(d) * xmp * ymp * h2) / (1 + P(e) * h2);
        Vnext(y) = (ymp - xmp * Vnext(z) * h2 + Vnext(z) * h2) / (1 - h2 * P(c));
        Vnext(x) = (xmp + h2 * Vnext(y) + P(b) * Vnext(y) * Vnext(z) * h2) / (1 + P(a) * h2);
    }
}