#include "main.h"
#include "fourwing.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, symmetry, method, COUNT };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_fourwing(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_fourwing);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_fourwing(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_fourwing, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_fourwing, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBSCAN_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4),
            H);
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_fourwing, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__ void finiteDifferenceScheme_fourwing(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + H * (P(a) * V(x) + V(y) * V(z));
        Vnext(y) = V(y) + H * (P(b) * V(x) + P(c) * V(y) - V(x) * V(z));
        Vnext(z) = V(z) + H * (-V(z) - V(x) * V(y));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * 0.5 * (P(a) * V(x) + V(y) * V(z));
        numb ymp = V(y) + H * 0.5 * (P(b) * V(x) + P(c) * V(y) - V(x) * V(z));
        numb zmp = V(z) + H * 0.5 * (-V(z) - V(x) * V(y));

        Vnext(x) = V(x) + H * (P(a) * xmp + ymp * zmp);
        Vnext(y) = V(y) + H * (P(b) * xmp + P(c) * ymp - xmp * zmp);
        Vnext(z) = V(z) + H * (-zmp - xmp * ymp);
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(a) * V(x) + V(y) * V(z);
        numb ky1 = P(b) * V(x) + P(c) * V(y) - V(x) * V(z);
        numb kz1 = -V(z) - V(x) * V(y);

        numb xmp = V(x) + 0.5 * H * kx1;
        numb ymp = V(y) + 0.5 * H * ky1;
        numb zmp = V(z) + 0.5 * H * kz1;

        numb kx2 = P(a) * xmp + ymp * zmp;
        numb ky2 = P(b) * xmp + P(c) * ymp - V(x) * zmp;
        numb kz2 = -zmp - xmp * ymp;

        xmp = V(x) + 0.5 * H * kx2;
        ymp = V(y) + 0.5 * H * ky2;
        zmp = V(z) + 0.5 * H * kz2;

        numb kx3 = P(a) * xmp + ymp * zmp;
        numb ky3 = P(b) * xmp + P(c) * ymp - V(x) * zmp;
        numb kz3 = -zmp - xmp * ymp;

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = P(a) * xmp + ymp * zmp;
        numb ky4 = P(b) * xmp + P(c) * ymp - V(x) * zmp;
        numb kz4 = -zmp - xmp * ymp;

        Vnext(x) = V(x) + H * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + H * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + H * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (P(a) * V(x) + V(y) * V(z));
        numb ymp = V(y) + h1 * (P(b) * xmp + P(c) * V(y) - xmp * V(z));
        numb zmp = V(z) + h1 * (-V(z) - xmp * ymp);

        Vnext(z) = (zmp - h2 * xmp * ymp) / (1 + h2);
        Vnext(y) = (ymp + h2 * P(b) * xmp - h2 * xmp * Vnext(z)) / (1 - h2 * P(c));
        Vnext(x) = (xmp + Vnext(y) * Vnext(z) * h2) / (1 - P(a) * h2);
    }
}