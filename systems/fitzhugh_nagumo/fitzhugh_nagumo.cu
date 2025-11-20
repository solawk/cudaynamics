#include "main.h"
#include "fitzhugh_nagumo.h"

#define name fitzhugh_nagumo

namespace attributes
{
    enum variables { v, w };
    enum parameters { a, b, tau, R, Iext, method, COUNT };
    enum methods { ExplicitEuler, ExplicitMidpoint };
}

__global__ void gpu_wrapper_(name)(Computation* data, uint64_t variation)
{
    kernelProgram_(name)(data, (blockIdx.x * blockDim.x) + threadIdx.x);
}

__host__ __device__ void kernelProgram_(name)(Computation* data, uint64_t variation)
{
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    uint64_t stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
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

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters)
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

#undef name