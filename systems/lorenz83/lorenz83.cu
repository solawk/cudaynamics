#include "main.h"
#include "lorenz83.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, f, g, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_lorenz83(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES;

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_lorenz83);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_lorenz83(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_lorenz83, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_lorenz83, MO(MAX));
    }
    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4), P(stepsize));
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_rossler, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__ void finiteDifferenceScheme_lorenz83(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (-P(a) * V(x) - pow(V(y), 2) - pow(V(z), 2) + P(a) * P(f));
        Vnext(y) = V(y) + P(stepsize) * (-V(y) + V(x) * V(y) - P(b) * V(x) * V(z) + P(g));
        Vnext(z) = V(z) + P(stepsize) * (-V(z) + P(b) * V(x) * V(y) + V(x) * V(z));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + P(stepsize) * 0.5 * (-P(a) * V(x) - pow(V(y), 2) - pow(V(z), 2) + P(a) * P(f));
        numb ymp = V(y) + P(stepsize) * 0.5 * (-V(y) + V(x) * V(y) - P(b) * V(x) * V(z) + P(g));
        numb zmp = V(z) + P(stepsize) * 0.5 * (-V(z) + P(b) * V(x) * V(y) + V(x) * V(z));

        Vnext(x) = V(x) + P(stepsize) * (-P(a) * xmp - pow(ymp, 2) - pow(zmp, 2) + P(a) * P(f));
        Vnext(y) = V(y) + P(stepsize) * (-ymp + xmp * ymp - P(b) * xmp * zmp + P(g));
        Vnext(z) = V(z) + P(stepsize) * (-zmp + P(b) * xmp * ymp + xmp * zmp);
    }
    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = -P(a) * V(x) - pow(V(y), 2) - pow(V(z), 2) + P(a) * P(f);
        numb ky1 = -V(y) + V(x) * V(y) - P(b) * V(x) * V(z) + P(g);
        numb kz1 = -V(z) + P(b) * V(x) * V(y) + V(x) * V(z);

        numb xmp = V(x) + 0.5 * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5 * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5 * P(stepsize) * kz1;

        numb kx2 = -P(a) * xmp - pow(ymp, 2) - pow(zmp, 2) + P(a) * P(f);
        numb ky2 = -ymp + xmp * ymp - P(b) * xmp * zmp + P(g);
        numb kz2 = -zmp + P(b) * xmp * ymp + xmp * zmp;

        xmp = V(x) + 0.5 * P(stepsize) * kx2;
        ymp = V(y) + 0.5 * P(stepsize) * ky2;
        zmp = V(z) + 0.5 * P(stepsize) * kz2;

        numb kx3 = -P(a) * xmp - pow(ymp, 2) - pow(zmp, 2) + P(a) * P(f);
        numb ky3 = -ymp + xmp * ymp - P(b) * xmp * zmp + P(g);
        numb kz3 = -zmp + P(b) * xmp * ymp + xmp * zmp;

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = -P(a) * xmp - pow(ymp, 2) - pow(zmp, 2) + P(a) * P(f);
        numb ky4 = -ymp + xmp * ymp - P(b) * xmp * zmp + P(g);
        numb kz4 = -zmp + P(b) * xmp * ymp + xmp * zmp;

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4) / 6.0;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4) / 6.0;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4) / 6.0;
    }
    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * P(stepsize) - P(symmetry);
        numb h2 = 0.5 * P(stepsize) + P(symmetry);

        numb xmp = V(x) + 0.5 * h1 * (-P(a) * V(x) - pow(V(y), 2) - pow(V(z), 2) + P(a) * P(f));
        numb ymp = V(y) + 0.5 * h1 * (-V(y) + xmp * V(y) - P(b) * xmp * V(z) + P(g));
        numb zmp = V(z) + 0.5 * h1 * (-V(z) + P(b) * xmp * ymp + xmp * V(z));

        numb denom_z = (1 + h2 - h2 * xmp);
        if (fabs(denom_z) < 1e-6) denom_z = copysign(1e-6, denom_z);
        Vnext(z) = (zmp + h2 * P(b) * xmp * ymp) / denom_z;

        numb denom_y = (1 + h2 - h2 * xmp);
        if (fabs(denom_y) < 1e-6) denom_y = copysign(1e-6, denom_y);
        Vnext(y) = (ymp - h2 * P(b) * xmp * Vnext(z) + h2 * P(g)) / denom_y;

        numb denom_x = (1 + h2 * P(a));
        if (fabs(denom_x) < 1e-6) denom_x = copysign(1e-6, denom_x);
        Vnext(x) = (xmp - h2 * pow(Vnext(y), 2) - h2 * pow(Vnext(z), 2) + h2 * P(a) * P(f)) / denom_x;
    }
}