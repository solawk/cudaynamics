#include "main.h"
#include "jj_mrlcs.h"

namespace attributes
{
    enum variables { theta, sin_theta, v, iL };
    enum parameters { betaL, betaC, betaM, i, epsilon, delta, symmetry, method };
    enum methods { ExplicitEuler, SemiExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_jj_mrlcs(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_jj_mrlcs);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_jj_mrlcs(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_jj_mrlcs, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_jj_mrlcs, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4), H);
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_jj_mrlcs, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__  void finiteDifferenceScheme_jj_mrlcs(numb* currentV, numb* nextV, numb* parameters, Computation* data)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(theta) = fmodf(V(theta) + H * V(v), 2.0f * 3.141592f);
        Vnext(sin_theta) = sinf(Vnext(theta));
        Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (V(v) - V(iL)));
        Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));
    }

    ifMETHOD(P(method), SemiExplicitEuler)
    {
        Vnext(theta) = fmodf(V(theta) + H * V(v), 2.0f * 3.141592f);
        Vnext(sin_theta) = sinf(Vnext(theta));
        Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (V(v) - V(iL)));
        Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(Vnext(theta))) * V(v) - sinf(Vnext(theta)) - P(delta) * Vnext(iL)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb thetamp = fmodf(V(theta) + H * 0.5 * V(v), 2.0f * 3.141592f);
        numb iLmp = V(iL) + H * 0.5 * ((1.0f / P(betaL)) * (V(v) - V(iL)));
        numb vmp = V(v) + H * 0.5 * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));

        Vnext(theta) = fmodf(V(theta) + H * vmp, 2.0f * 3.141592f);
        Vnext(sin_theta) = sinf(Vnext(theta));
        Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (vmp - iLmp));
        Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb ktheta1 = V(v);
        numb kiL1 = ((1.0f / P(betaL)) * (V(v) - V(iL)));
        numb kv1 = ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));

        numb thetamp = fmodf(V(theta) + H * 0.5 * ktheta1, 2.0f * 3.141592f);
        numb iLmp = V(iL) + H * 0.5 * kiL1;
        numb vmp = V(v) + H * 0.5 * kv1;

        numb ktheta2 = vmp;
        numb kiL2 = ((1.0f / P(betaL)) * (vmp - iLmp));
        numb kv2 = ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

        thetamp = fmodf(V(theta) + H * 0.5 * ktheta2, 2.0f * 3.141592f);
        iLmp = V(iL) + H * 0.5 * kiL2;
        vmp = V(v) + H * 0.5 * kv2;

        numb ktheta3 = vmp;
        numb kiL3 = ((1.0f / P(betaL)) * (vmp - iLmp));
        numb kv3 = ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

        thetamp = fmodf(V(theta) + H * ktheta3, 2.0f * 3.141592f);
        iLmp = V(iL) + H * kiL3;
        vmp = V(v) + H * kv3;

        numb ktheta4 = vmp;
        numb kiL4 = ((1.0f / P(betaL)) * (vmp - iLmp));
        numb kv4 = ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

        Vnext(theta) = fmodf(V(theta) + H * (ktheta1 + 2 * ktheta2 + 2 * ktheta3 + ktheta4) / 6, 2.0f * 3.141592f);
        Vnext(sin_theta) = sinf(Vnext(theta));
        Vnext(iL) = V(iL) + H * (kiL1 + 2 * kiL2 + 2 * kiL3 + kiL4) / 6;
        Vnext(v) = V(v) + H * (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb thetamp = fmodf(V(theta) + h1 * V(v), 2.0f * 3.141592f);
        numb iLmp = V(iL) + h1 * ((1.0f / P(betaL)) * (V(v) - V(iL)));
        numb vmp = V(v) + h1 * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * V(v) - sinf(thetamp) - P(delta) * iLmp));

        Vnext(v) = vmp + h2 * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));
        Vnext(v) = vmp + h2 * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * Vnext(v) - sinf(thetamp) - P(delta) * iLmp));
        Vnext(iL) = (iLmp + h2 * (1.0f / P(betaL)) * Vnext(v)) / (1 + h2 * (1.0f / P(betaL)));
        Vnext(theta) = fmodf(thetamp + h2 * Vnext(v), 2.0f * 3.141592f);
        Vnext(sin_theta) = sinf(Vnext(theta));
    }
}