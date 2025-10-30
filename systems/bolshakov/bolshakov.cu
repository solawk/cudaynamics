#include "main.h"
#include "bolshakov.h"

namespace attributes
{
    enum variables { Q, S, X, Y, c };
    enum parameters { i, p, k, r, COUNT };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_bolshakov(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_bolshakov);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_bolshakov(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_bolshakov, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_bolshakov, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBSCAN_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4),
            H);
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_bolshakov, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__  __forceinline__ void finiteDifferenceScheme_bolshakov(numb* currentV, numb* nextV, numb* parameters)
{
    Vnext(Q) = V(Q) + V(c) * (P(i) + V(S)) - !V(c) * (P(i) + V(S));

    if (Vnext(Q) < 0) Vnext(Q) = 0;
    if (Vnext(Q) > P(r)) Vnext(Q) = P(r);

    Vnext(X) = P(p) * (Vnext(Q) + V(X));
    Vnext(Y) = 1/P(k) * (Vnext(X) - V(Y)) + V(Y);
    Vnext(S) = Vnext(X) - Vnext(Y);
    Vnext(c) = (V(c) || (Vnext(Q) > 0 ? 0 : 1)) && (Vnext(Q) < P(r) ? 1 : 0);
}