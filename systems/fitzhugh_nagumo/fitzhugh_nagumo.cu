#include "main.h"
#include "fitzhugh_nagumo.h"

namespace attributes
{
    enum variables { v, w };
    enum parameters { a, b, tau, R, Iext,method };
    enum methods { ExplicitEuler, ExplicitMidpoint };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_fitzhugh_nagumo(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_fitzhugh_nagumo);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_fitzhugh_nagumo(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_fitzhugh_nagumo, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_fitzhugh_nagumo, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4),
            H_BRANCH(parameters[CUDA_kernel.PARAM_COUNT - 1], variables[CUDA_kernel.VAR_COUNT - 1]));
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_fitzhugh_nagumo, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__  void finiteDifferenceScheme_fitzhugh_nagumo(numb* currentV, numb* nextV, numb* parameters, Computation* data)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(v) = V(v) + H * (V(v) - (V(v) * V(v) * V(v) / 3.0f) - V(w) + P(R) * P(Iext));
        Vnext(w) = V(w) + H * ((V(v) + P(a) - P(b) * V(w)) / P(tau));
    }
    
    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb vmp = V(v) + H * 0.5f * (V(v) - (V(v) * V(v) * V(v) / 3.0f) - V(w) + P(R) * P(Iext));
        numb wmp = V(w) + H * 0.5f * ((V(v) + P(a) - P(b) * V(w)) / P(tau));

        Vnext(v) = V(v) + H * (vmp - (vmp * vmp * vmp / 3.0f) - wmp + P(R) * P(Iext));
        Vnext(w) = V(w) + H * ((vmp + P(a) - P(b) * wmp) / P(tau));
    }
}