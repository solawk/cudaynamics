#include "mishchenko.h"

#define name mishchenko

namespace attributes
{
    enum variables { x, cosx, y, z, gamma, t };
    enum parameters { eps1, eps2, Gammadc, Gammaamp, Gammafreq, Gammadel, Gammadf, signal, COUNT };
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
        numb gamp = P(Gammadc) + (fmod((V(t) - P(Gammadel)) > (numb)0.0 ? (V(t) - P(Gammadel)) : (P(Gammadf) / P(Gammafreq) + P(Gammadel) - V(t)), (numb)1.0 / P(Gammafreq)) < P(Gammadf) / P(Gammafreq) ? P(Gammaamp) : (numb)0.0);
        numb tmp = V(t) + H * (numb)0.5;

        numb xmp = fmod(V(x) + H * (numb)0.5 * V(y), (numb)2.0 * (numb)3.141592);
        numb ymp = V(y) + H * (numb)0.5 * V(z);
        numb zmp = V(z) + H * (numb)0.5 * (((numb)1.0 / (P(eps1) * P(eps2))) * (gamp - (P(eps1) + P(eps2)) * V(z) - ((numb)1.0 - (P(eps1) * cos(V(x)))) * V(y)));

        Vnext(gamma) = P(Gammadc) + (fmod((tmp - P(Gammadel)) > (numb)0.0 ? (tmp - P(Gammadel)) : (P(Gammadf) / P(Gammafreq) + P(Gammadel) - tmp), (numb)1.0 / P(Gammafreq)) < P(Gammadf) / P(Gammafreq) ? P(Gammaamp) : (numb)0.0);
        Vnext(t) = V(t) + H;

        Vnext(x) = fmod(V(x) + H * ymp, (numb)2.0 * (numb)3.141592);
        Vnext(cosx) = cos(Vnext(x));
        Vnext(y) = V(y) + H * zmp;
        Vnext(z) = V(z) + H * (numb)0.5 * (((numb)1.0 / (P(eps1) * P(eps2))) * (Vnext(gamma) - (P(eps1) + P(eps2)) * zmp - ((numb)1.0 - (P(eps1) * cos(xmp))) * ymp));
    }

    ifSIGNAL(P(signal), sine)
    {
        numb gamp = P(Gammadc) + P(Gammaamp) * sin((numb)2.0 * (numb)3.141592 * P(Gammafreq) * (V(t) - P(Gammadel)));
        numb tmp = V(t) + H * (numb)0.5;

        numb xmp = fmod(V(x) + H * (numb)0.5 * V(y), (numb)2.0 * (numb)3.141592);
        numb ymp = V(y) + H * (numb)0.5 * V(z);
        numb zmp = V(z) + H * (numb)0.5 * (((numb)1.0 / (P(eps1) * P(eps2))) * (gamp - (P(eps1) + P(eps2)) * V(z) - ((numb)1.0 - (P(eps1) * cos(V(x)))) * V(y)));

        Vnext(gamma) = P(Gammadc) + P(Gammaamp) * sin((numb)2.0 * (numb)3.141592 * P(Gammafreq) * (tmp - P(Gammadel)));
        Vnext(t) = V(t) + H;

        Vnext(x) = fmod(V(x) + H * ymp, (numb)2.0 * (numb)3.141592);
        Vnext(cosx) = cos(Vnext(x));
        Vnext(y) = V(y) + H * zmp;
        Vnext(z) = V(z) + H * (numb)0.5 * (((numb)1.0 / (P(eps1) * P(eps2))) * (Vnext(gamma) - (P(eps1) + P(eps2)) * zmp - ((numb)1.0 - (P(eps1) * cos(xmp))) * ymp));
    }

    ifSIGNAL(P(signal), triangle)
    {
        numb gamp = P(Gammadc) + P(Gammaamp) * (((numb)4.0 * P(Gammafreq) * (V(t) - P(Gammadel)) - (numb)2.0 * floor((((numb)4.0 * P(Gammafreq) * (V(t) - P(Gammadel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Gammafreq) * (V(t) - P(Gammadel)) + (numb)1.0) / (numb)2.0))));
        numb tmp = V(t) + H * (numb)0.5;

        numb xmp = fmod(V(x) + H * (numb)0.5 * V(y), (numb)2.0 * (numb)3.141592);
        numb ymp = V(y) + H * (numb)0.5 * V(z);
        numb zmp = V(z) + H * (numb)0.5 * (((numb)1.0 / (P(eps1) * P(eps2))) * (gamp - (P(eps1) + P(eps2)) * V(z) - ((numb)1.0 - (P(eps1) * cos(V(x)))) * V(y)));

        Vnext(gamma) = P(Gammadc) + P(Gammaamp) * (((numb)4.0 * P(Gammafreq) * (tmp - P(Gammadel)) - (numb)2.0 * floor((((numb)4.0 * P(Gammafreq) * (tmp - P(Gammadel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Gammafreq) * (tmp - P(Gammadel)) + (numb)1.0) / (numb)2.0))));
        Vnext(t) = V(t) + H;

        Vnext(x) = fmod(V(x) + H * ymp, (numb)2.0 * (numb)3.141592);
        Vnext(cosx) = cos(Vnext(x));
        Vnext(y) = V(y) + H * zmp;
        Vnext(z) = V(z) + H * (numb)0.5 * (((numb)1.0 / (P(eps1) * P(eps2))) * (Vnext(gamma) - (P(eps1) + P(eps2)) * zmp - ((numb)1.0 - (P(eps1) * cos(xmp))) * ymp));
    }

}

#undef name