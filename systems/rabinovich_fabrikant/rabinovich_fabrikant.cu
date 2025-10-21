#include "main.h"
#include "rabinovich_fabrikant.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { alpha, betta, gamma, delta, epsilon, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_rabinovich_fabrikant(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_rabinovich_fabrikant);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_rabinovich_fabrikant(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_rabinovich_fabrikant, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_rabinovich_fabrikant, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4), P(stepsize));
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_rabinovich_fabrikant, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__ void finiteDifferenceScheme_rabinovich_fabrikant(numb* currentV, numb* nextV, numb* parameters, Computation* data)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (V(y) * (V(z) - P(betta) + pow(V(x), 2)) + P(gamma) * V(x));
        Vnext(y) = V(y) + P(stepsize) * (V(x) * (P(delta) * V(z) + P(betta) - pow(V(x), 2)) + P(gamma) * V(y));
        Vnext(z) = V(z) + P(stepsize) * (P(epsilon) * V(z) * (P(alpha) + V(x) * V(y)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + P(stepsize) * 0.5 * (V(y) * (V(z) - P(betta) + pow(V(x), 2)) + P(gamma) * V(x));
        numb ymp = V(y) + P(stepsize) * 0.5 * (V(x) * (P(delta) * V(z) + P(betta) - pow(V(x), 2)) + P(gamma) * V(y));
        numb zmp = V(z) + P(stepsize) * 0.5 * (P(epsilon) * V(z) * (P(alpha) + V(x) * V(y)));

        Vnext(x) = V(x) + P(stepsize) * (ymp * (zmp - P(betta) + pow(xmp, 2)) + P(gamma) * xmp);
        Vnext(y) = V(y) + P(stepsize) * (xmp * (P(delta) * zmp + P(betta) - pow(xmp, 2)) + P(gamma) * ymp);
        Vnext(z) = V(z) + P(stepsize) * (P(epsilon) * zmp * (P(alpha) + xmp * ymp));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = V(y) * (V(z) - P(betta) + pow(V(x), 2)) + P(gamma) * V(x);
        numb ky1 = V(x) * (P(delta) * V(z) + P(betta) - pow(V(x), 2)) + P(gamma) * V(y);
        numb kz1 = P(epsilon) * V(z) * (P(alpha) + V(x) * V(y));

        numb xmp = V(x) + 0.5 * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5 * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5 * P(stepsize) * kz1;

        numb kx2 = ymp * (zmp - P(betta) + pow(xmp, 2)) + P(gamma) * xmp;
        numb ky2 = xmp * (P(delta) * zmp + P(betta) - pow(xmp, 2)) + P(gamma) * ymp;
        numb kz2 = P(epsilon) * zmp * (P(alpha) + xmp * ymp);

        xmp = V(x) + 0.5 * P(stepsize) * kx2;
        ymp = V(y) + 0.5 * P(stepsize) * ky2;
        zmp = V(z) + 0.5 * P(stepsize) * kz2;

        numb kx3 = ymp * (zmp - P(betta) + pow(xmp, 2)) + P(gamma) * xmp;
        numb ky3 = xmp * (P(delta) * zmp + P(betta) - pow(xmp, 2)) + P(gamma) * ymp;
        numb kz3 = P(epsilon) * zmp * (P(alpha) + xmp * ymp);

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = ymp * (zmp - P(betta) + pow(xmp, 2)) + P(gamma) * xmp;
        numb ky4 = xmp * (P(delta) * zmp + P(betta) - pow(xmp, 2)) + P(gamma) * ymp;
        numb kz4 = P(epsilon) * zmp * (P(alpha) + xmp * ymp);

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4) / 6.0;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4) / 6.0;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4) / 6.0;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * P(stepsize) - P(symmetry);
        numb h2 = 0.5 * P(stepsize) + P(symmetry);

        numb xmp = V(x) + h1 * (V(y) * (V(z) - P(betta) + pow(V(x), 2)) + P(gamma) * V(x));
        numb ymp = V(y) + h1 * (xmp * (P(delta) * V(z) + P(betta) - pow(xmp, 2)) + P(gamma) * V(y));
        numb zmp = V(z) + h1 * (P(epsilon) * V(z) * (P(alpha) + xmp * ymp));

        numb denom_z = 1.0 - h2 * P(epsilon) * (P(alpha) + xmp * ymp);
        if (fabs(denom_z) < 1e-6) denom_z = copysign(1e-6, denom_z);
        Vnext(z) = zmp / denom_z;

        numb denom_y = 1.0 - h2 * P(gamma);
        if (fabs(denom_y) < 1e-6) denom_y = copysign(1e-6, denom_y);
        Vnext(y) = (ymp + h2 * xmp * (P(delta) * Vnext(z) + P(betta) - pow(xmp, 2))) / denom_y;

        Vnext(x) = xmp + h2 * (Vnext(y) * (Vnext(z) - P(betta) + pow(xmp, 2)) + P(gamma) * xmp);
        Vnext(x) = xmp + h2 * (Vnext(y) * (Vnext(z) - P(betta) + pow(Vnext(x), 2)) + P(gamma) * Vnext(x));
    }
}
