#include "main.h"
#include "three_scroll.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, d, e, f, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_three_scroll(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_three_scroll);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_three_scroll(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_three_scroll, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_three_scroll, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4),
            H_BRANCH(parameters[CUDA_kernel.PARAM_COUNT - 1], variables[CUDA_kernel.VAR_COUNT - 1]));
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_three_scroll, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__  void finiteDifferenceScheme_three_scroll(numb* currentV, numb* nextV, numb* parameters, Computation* data)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + H * (P(a) * (V(y) - V(x)) + P(d) * V(x) * V(z));
        Vnext(y) = V(y) + H * (V(x) * (P(b) - V(z)) + P(f) * V(y));
        Vnext(z) = V(z) + H * (P(c) * V(z) + V(x) * (V(y) - P(e) * V(x)));
    }


    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + 0.5 * H * (P(a) * (V(y) - V(x)) + P(d) * V(x) * V(z));
        numb ymp = V(y) + 0.5 * H * (V(x) * (P(b) - V(z)) + P(f) * V(y));
        numb zmp = V(z) + 0.5 * H * (P(c) * V(z) + V(x) * (V(y) - P(e) * V(x)));

        Vnext(x) = V(x) + H * (P(a) * (ymp - xmp) + P(d) * xmp * zmp);
        Vnext(y) = V(y) + H * (xmp * (P(b) - zmp) + P(f) * ymp);
        Vnext(z) = V(z) + H * (P(c) * zmp + xmp * (ymp - P(e) * xmp));
    }


    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(a) * (V(y) - V(x)) + P(d) * V(x) * V(z);
        numb ky1 = V(x) * (P(b) - V(z)) + P(f) * V(y);
        numb kz1 = P(c) * V(z) + V(x) * (V(y) - P(e) * V(x));

        numb xmp = V(x) + 0.5 * H * kx1;
        numb ymp = V(y) + 0.5 * H * ky1;
        numb zmp = V(z) + 0.5 * H * kz1;

        numb kx2 = P(a) * (ymp - xmp) + P(d) * xmp * zmp;
        numb ky2 = xmp * (P(b) - zmp) + P(f) * ymp;
        numb kz2 = P(c) * zmp + xmp * (ymp - P(e) * xmp);

        xmp = V(x) + 0.5 * H * kx2;
        ymp = V(y) + 0.5 * H * ky2;
        zmp = V(z) + 0.5 * H * kz2;

        numb kx3 = P(a) * (ymp - xmp) + P(d) * xmp * zmp;
        numb ky3 = xmp * (P(b) - zmp) + P(f) * ymp;
        numb kz3 = P(c) * zmp + xmp * (ymp - P(e) * xmp);

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = P(a) * (ymp - xmp) + P(d) * xmp * zmp;
        numb ky4 = xmp * (P(b) - zmp) + P(f) * ymp;
        numb kz4 = P(c) * zmp + xmp * (ymp - P(e) * xmp);

        Vnext(x) = V(x) + H * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + H * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + H * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }


    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (P(a) * (V(y) - V(x)) + P(d) * V(x) * V(z));
        numb ymp = V(y) + h1 * (xmp * (P(b) - V(z)) + P(f) * V(y));
        numb zmp = V(z) + h1 * (P(c) * V(z) + xmp * (ymp - P(e) * xmp));

        Vnext(z) = (zmp + h2 * xmp * (ymp - P(e) * xmp)) / (1 - h2 * P(c));
        Vnext(y) = (ymp + h2 * xmp * (P(b) - Vnext(z))) / (1 - h2 * P(f));
        Vnext(x) = (xmp + h2 * P(a) * Vnext(y)) / (1 + h2 * (P(a) - P(d) * Vnext(z)));
    }
}