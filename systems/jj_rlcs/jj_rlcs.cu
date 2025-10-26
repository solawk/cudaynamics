#include "main.h"
#include "jj_rlcs.h"

namespace attributes
{
    enum variables { x1, sin_x1, x2, x3 };
    enum parameters { betaL, betaC, i, gThreshold, Rn, Rsg, symmetry, method, COUNT };
    enum methods { ExplicitEuler, SemiExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_jj_rlcs(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_jj_rlcs);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_jj_rlcs(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_jj_rlcs, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_jj_rlcs, MO(MAX));
    }
    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBSCAN_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4),
            H);
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_jj_rlcs, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__ void finiteDifferenceScheme_jj_rlcs(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x1) = fmodf(V(x1) + H * V(x2), 2.0f * 3.141592f);
        Vnext(sin_x1) = sinf(Vnext(x1));
        Vnext(x2) = V(x2) + H * (1.0f / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sinf(V(x1)) - V(x3));
        Vnext(x3) = V(x3) + H * (1.0f / P(betaL)) * (V(x2) - V(x3));
    }

    ifMETHOD(P(method), SemiExplicitEuler)
    {
        Vnext(x1) = fmodf(V(x1) + H * V(x2), 2.0f * 3.141592f);
        Vnext(sin_x1) = sinf(Vnext(x1));
        Vnext(x3) = V(x3) + H * (1.0f / P(betaL)) * (V(x2) - V(x3));
        Vnext(x2) = V(x2) + H * (1.0f / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sinf(Vnext(x1)) - Vnext(x3));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb x1mp = fmodf(V(x1) + H * 0.5 * V(x2), 2.0f * 3.141592f);
        numb x2mp = V(x2) + H * 0.5 * (1.0f / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sinf(V(x1)) - V(x3));
        numb x3mp = V(x3) + H * 0.5 * (1.0f / P(betaL)) * (V(x2) - V(x3));

        Vnext(x1) = fmodf(V(x1) + H * x2mp, 2.0f * 3.141592f);
        Vnext(sin_x1) = sinf(Vnext(x1));
        Vnext(x2) = V(x2) + H * (1.0f / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sinf(x1mp) - x3mp);
        Vnext(x3) = V(x3) + H * (1.0f / P(betaL)) * (x2mp - x3mp);
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx11 = V(x2);
        numb kx21 = (1.0f / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sinf(V(x1)) - V(x3));
        numb kx31 = (1.0f / P(betaL)) * (V(x2) - V(x3));

        numb x1mp = fmodf(V(x1) + H * 0.5 * kx11, 2.0f * 3.141592f);
        numb x2mp = V(x2) + H * 0.5 * kx21;
        numb x3mp = V(x3) + H * 0.5 * kx31;

        numb kx12 = x2mp;
        numb kx22 = (1.0f / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sinf(x1mp) - x3mp);
        numb kx32 = (1.0f / P(betaL)) * (x2mp - x3mp);

        x1mp = fmodf(V(x1) + H * 0.5 * kx12, 2.0f * 3.141592f);
        x2mp = V(x2) + H * 0.5 * kx22;
        x3mp = V(x3) + H * 0.5 * kx32;

        numb kx13 = x2mp;
        numb kx23 = (1.0f / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sinf(x1mp) - x3mp);
        numb kx33 = (1.0f / P(betaL)) * (x2mp - x3mp);

        x1mp = fmodf(V(x1) + H * kx13, 2.0f * 3.141592f);
        x2mp = V(x2) + H * kx23;
        x3mp = V(x3) + H * kx33;

        numb kx14 = x2mp;
        numb kx24 = (1.0f / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sinf(x1mp) - x3mp);
        numb kx34 = (1.0f / P(betaL)) * (x2mp - x3mp);

        Vnext(x1) = fmodf(V(x1) + H * (kx11 + 2 * kx12 + 2 * kx13 + kx14) / 6, 2.0f * 3.141592f);
        Vnext(sin_x1) = sinf(Vnext(x1));
        Vnext(x2) = V(x2) + H * (kx21 + 2 * kx22 + 2 * kx23 + kx24) / 6;
        Vnext(x3) = V(x3) + H * (kx31 + 2 * kx32 + 2 * kx33 + kx34) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb x1mp = fmodf(V(x1) + h1 * V(x2), 2.0f * 3.141592f);
        numb x3mp = V(x3) + h1 * (1.0f / P(betaL)) * (V(x2) - V(x3));
        numb x2mp = V(x2) + h1 * (1.0f / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sinf(x1mp) - x3mp);

        Vnext(x2) = x2mp + h2 * (1.0f / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sinf(x1mp) - x3mp);
        Vnext(x2) = x2mp + h2 * (1.0f / P(betaC)) * (P(i) - ((Vnext(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * Vnext(x2) - sinf(x1mp) - x3mp);
        Vnext(x3) = (x3mp + h2 * (1.0f / P(betaL)) * Vnext(x2)) / (1.0f + h2 * (1.0f / P(betaL)));
        Vnext(x1) = fmodf(x1mp + h2 * Vnext(x2), 2.0f * 3.141592f);
        Vnext(sin_x1) = sinf(Vnext(x1));
    }
}
