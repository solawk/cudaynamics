#include "bolshakov.h"

#define name bolshakov

namespace attributes
{
    enum variables { Q, S, X, Y, c, i, t };
    enum parameters { p, k, r, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, COUNT };
    enum waveforms { square, sine, triangle };
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
    ifSIGNAL(P(signal), square)
    {
        Vnext(i) = P(Idc) + (std::fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
        Vnext(t) = V(t) + H;
        Vnext(Q) = V(Q) + V(c) * (Vnext(i) + V(S)) - !V(c) * (Vnext(i) + V(S));

        if (Vnext(Q) < (numb)0) Vnext(Q) = (numb)0;
        if (Vnext(Q) > P(r)) Vnext(Q) = P(r);

        Vnext(X) = P(p) * (Vnext(Q) + V(X));
        Vnext(Y) = (numb)1 / P(k) * (Vnext(X) - V(Y)) + V(Y);
        Vnext(S) = Vnext(X) - Vnext(Y);
        Vnext(c) = (V(c) || (Vnext(Q) > (numb)0 ? (numb)0 : (numb)1)) && (Vnext(Q) < P(r) ? (numb)1 : (numb)0);
    }
    ifSIGNAL(P(signal), sine)
    {
        Vnext(i) = P(Idc) + P(Iamp) * std::sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
        Vnext(t) = V(t) + H;
        Vnext(Q) = V(Q) + V(c) * (Vnext(i) + V(S)) - !V(c) * (Vnext(i) + V(S));

        if (Vnext(Q) < (numb)0) Vnext(Q) = (numb)0;
        if (Vnext(Q) > P(r)) Vnext(Q) = P(r);

        Vnext(X) = P(p) * (Vnext(Q) + V(X));
        Vnext(Y) = (numb)1 / P(k) * (Vnext(X) - V(Y)) + V(Y);
        Vnext(S) = Vnext(X) - Vnext(Y);
        Vnext(c) = (V(c) || (Vnext(Q) > (numb)0 ? (numb)0 : (numb)1)) && (Vnext(Q) < P(r) ? (numb)1 : (numb)0);
    }
    ifSIGNAL(P(signal), triangle)
    {
        Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * std::floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)) * std::pow((numb)(-1), std::floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)));
        Vnext(t) = V(t) + H;
        Vnext(Q) = V(Q) + V(c) * (Vnext(i) + V(S)) - !V(c) * (Vnext(i) + V(S));

        if (Vnext(Q) < (numb)0) Vnext(Q) = (numb)0;
        if (Vnext(Q) > P(r)) Vnext(Q) = P(r);

        Vnext(X) = P(p) * (Vnext(Q) + V(X));
        Vnext(Y) = (numb)1 / P(k) * (Vnext(X) - V(Y)) + V(Y);
        Vnext(S) = Vnext(X) - Vnext(Y);
        Vnext(c) = (V(c) || (Vnext(Q) > (numb)0 ? (numb)0 : (numb)1)) && (Vnext(Q) < P(r) ? (numb)1 : (numb)0);
    }
}

#undef name