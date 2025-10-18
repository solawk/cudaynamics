#include "main.h"
#include "sang.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, mu, omega, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_sang(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES;

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_sang);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_sang(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_sang, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_sang, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4), P(stepsize));
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_rossler, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}
__device__ __forceinline__ void finiteDifferenceScheme_sang(numb* currentV, numb* nextV, numb* parameters, Computation* data)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (-V(y));
        Vnext(y) = V(y) + P(stepsize) * (V(x) + P(c) * V(y) + P(a) * V(z));
        Vnext(z) = V(z) + P(stepsize) * (-P(mu) * V(z) + P(b) * cosf(P(omega) * V(y)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + P(stepsize) * 0.5 * (-V(y));
        numb ymp = V(y) + P(stepsize) * 0.5 * (V(x) + P(c) * V(y) + P(a) * V(z));
        numb zmp = V(z) + P(stepsize) * 0.5 * (-P(mu) * V(z) + P(b) * cosf(P(omega) * V(y)));

        Vnext(x) = V(x) + P(stepsize) * (-ymp);
        Vnext(y) = V(y) + P(stepsize) * (xmp + P(c) * ymp + P(a) * zmp);
        Vnext(z) = V(z) + P(stepsize) * (-P(mu) * zmp + P(b) * cosf(P(omega) * ymp));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = -V(y);
        numb ky1 = V(x) + P(c) * V(y) + P(a) * V(z);
        numb kz1 = -P(mu) * V(z) + P(b) * cosf(P(omega) * V(y));

        numb xmp = V(x) + 0.5 * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5 * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5 * P(stepsize) * kz1;

        numb kx2 = -ymp;
        numb ky2 = xmp + P(c) * ymp + P(a) * zmp;
        numb kz2 = -P(mu) * zmp + P(b) * cosf(P(omega) * ymp);

        xmp = V(x) + 0.5 * P(stepsize) * kx2;
        ymp = V(y) + 0.5 * P(stepsize) * ky2;
        zmp = V(z) + 0.5 * P(stepsize) * kz2;

        numb kx3 = -ymp;
        numb ky3 = xmp + P(c) * ymp + P(a) * zmp;
        numb kz3 = -P(mu) * zmp + P(b) * cosf(P(omega) * ymp);

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = -ymp;
        numb ky4 = xmp + P(c) * ymp + P(a) * zmp;
        numb kz4 = -P(mu) * zmp + P(b) * cosf(P(omega) * ymp);

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * P(stepsize) - P(symmetry);
        numb h2 = 0.5 * P(stepsize) + P(symmetry);

        numb xmp = V(x) + h1 * (-V(y));
        numb ymp = V(y) + h1 * (xmp + P(c) * V(y) + P(a) * V(z));
        numb zmp = V(z) + h1 * (-P(mu) * V(z) + P(b) * cosf(P(omega) * ymp));

        Vnext(z) = (zmp + P(b) * cosf(P(omega) * ymp) * h2) / (1 + P(mu) * h2);
        Vnext(y) = (ymp + (xmp + P(a) * Vnext(z)) * h2) / (1 - P(c) * h2);
        Vnext(x) = xmp + h2 * (-Vnext(y));
    }
}