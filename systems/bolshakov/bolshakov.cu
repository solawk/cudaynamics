#include "main.h"
#include "bolshakov.h"

#define name bolshakov

namespace attributes
{
    enum variables { Q, S, X, Y, c };
    enum parameters { i, p, k, r, COUNT };
}

__global__ void kernelProgram_(name)(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_(name));

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_(name)(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis
    AnalysisLobby(data, &finiteDifferenceScheme_(name), variation);
}

__device__  __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters)
{
    Vnext(Q) = V(Q) + V(c) * (P(i) + V(S)) - !V(c) * (P(i) + V(S));

    if (Vnext(Q) < 0) Vnext(Q) = 0;
    if (Vnext(Q) > P(r)) Vnext(Q) = P(r);

    Vnext(X) = P(p) * (Vnext(Q) + V(X));
    Vnext(Y) = 1/P(k) * (Vnext(X) - V(Y)) + V(Y);
    Vnext(S) = Vnext(X) - Vnext(Y);
    Vnext(c) = (V(c) || (Vnext(Q) > 0 ? 0 : 1)) && (Vnext(Q) < P(r) ? 1 : 0);
}

#undef name