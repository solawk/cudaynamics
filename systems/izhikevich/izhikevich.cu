#include "main.h"
#include "izhikevich.h"

namespace attributes
{
    enum variables { v, u, I, t };
    enum parameters { a, b, c, d, p0, p1, p2, p3, Imax, Idc, stepsize, method };
    enum methods { ExplicitEuler };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_izhikevich(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_izhikevich);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_izhikevich(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_izhikevich, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_izhikevich, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4), P(stepsize));
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_izhikevich, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__  void finiteDifferenceScheme_izhikevich(numb* currentV, numb* nextV, numb* parameters, Computation* data)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(I) = fmodf(V(t), P(Idc)) < (0.5f * P(Idc)) ? P(Imax) : 0.0f;
        Vnext(t) = V(t) + P(stepsize);
        Vnext(v) = V(v) + P(stepsize) * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(I));
        Vnext(u) = V(u) + P(stepsize) * (P(a) * (P(b) * V(v) - V(u)));

        if (Vnext(v) >= P(p3))
        {
            Vnext(v) = P(c);
            Vnext(u) = Vnext(u) + P(d);
        }
    }
}