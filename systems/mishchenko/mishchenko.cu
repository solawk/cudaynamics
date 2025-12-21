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
        numb gamp= P(Gammadc) + (fmodf((V(t) - P(Gammadel)) > 0 ? (V(t) - P(Gammadel)) : (P(Gammadf) / P(Gammafreq) + P(Gammadel) - V(t)), 1 / P(Gammafreq)) < P(Gammadf) / P(Gammafreq) ? P(Gammaamp) : 0.0f);
        numb tmp = V(t) + H*0.5;

        numb xmp = fmodf(V(x) + H * 0.5 * V(y), 2.0f * 3.141592f);
        numb ymp = V(y) + H * 0.5 * V(z);
        numb zmp = V(z) + H * 0.5 * ((1 / (P(eps1) * P(eps2))) * (gamp - (P(eps1) + P(eps2)) * V(z) - (1 - (P(eps1) * cosf(V(x)))) * V(y)));

        Vnext(gamma) = P(Gammadc) + (fmodf((tmp - P(Gammadel)) > 0 ? (tmp - P(Gammadel)) : (P(Gammadf) / P(Gammafreq) + P(Gammadel) - tmp), 1 / P(Gammafreq)) < P(Gammadf) / P(Gammafreq) ? P(Gammaamp) : 0.0f);
        Vnext(t) = V(t) + H;

        Vnext(x) = fmodf(V(x) + H * ymp, 2.0f * 3.141592f);
        Vnext(cosx) = cosf(Vnext(x));
        Vnext(y) = V(y) + H * zmp;
        Vnext(z) = V(z) + H * 0.5 * ((1 / (P(eps1) * P(eps2))) * (Vnext(gamma) - (P(eps1) + P(eps2)) * zmp - (1 - (P(eps1) * cosf(xmp))) * ymp));
    }

    ifSIGNAL(P(signal), sine)
    {
        numb gamp = P(Gammadc) + P(Gammaamp) * sinf(2.0f * 3.141592f * P(Gammafreq) * (V(t) - P(Gammadel)));
        numb tmp = V(t) + H * 0.5;

        numb xmp = fmodf(V(x) + H * 0.5 * V(y), 2.0f * 3.141592f);
        numb ymp = V(y) + H * 0.5 * V(z);
        numb zmp = V(z) + H * 0.5 * ((1 / (P(eps1) * P(eps2))) * (gamp - (P(eps1) + P(eps2)) * V(z) - (1 - (P(eps1) * cosf(V(x)))) * V(y)));

        Vnext(gamma) = P(Gammadc) + P(Gammaamp) * sinf(2.0f * 3.141592f * P(Gammafreq) * (tmp - P(Gammadel)));
        Vnext(t) = V(t) + H;

        Vnext(x) = fmodf(V(x) + H * ymp, 2.0f * 3.141592f);
        Vnext(cosx) = cosf(Vnext(x));
        Vnext(y) = V(y) + H * zmp;
        Vnext(z) = V(z) + H * 0.5 * ((1 / (P(eps1) * P(eps2))) * (Vnext(gamma) - (P(eps1) + P(eps2)) * zmp - (1 - (P(eps1) * cosf(xmp))) * ymp));
    }

    ifSIGNAL(P(signal), triangle)
    {
        numb gamp = P(Gammadc) + P(Gammaamp) * ((4.0f * P(Gammafreq) * (V(t) - P(Gammadel)) - 2.0f * floorf((4.0f * P(Gammafreq) * (V(t) - P(Gammadel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Gammafreq) * (V(t) - P(Gammadel)) + 1.0f) / 2.0f)));
        numb tmp = V(t) + H * 0.5;

        numb xmp = fmodf(V(x) + H * 0.5 * V(y), 2.0f * 3.141592f);
        numb ymp = V(y) + H * 0.5 * V(z);
        numb zmp = V(z) + H * 0.5 * ((1 / (P(eps1) * P(eps2))) * (gamp - (P(eps1) + P(eps2)) * V(z) - (1 - (P(eps1) * cosf(V(x)))) * V(y)));

        Vnext(gamma) = P(Gammadc) + P(Gammaamp) * ((4.0f * P(Gammafreq) * (tmp - P(Gammadel)) - 2.0f * floorf((4.0f * P(Gammafreq) * (tmp - P(Gammadel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Gammafreq) * (tmp - P(Gammadel)) + 1.0f) / 2.0f)));
        Vnext(t) = V(t) + H;

        Vnext(x) = fmodf(V(x) + H * ymp, 2.0f * 3.141592f);
        Vnext(cosx) = cosf(Vnext(x));
        Vnext(y) = V(y) + H * zmp;
        Vnext(z) = V(z) + H * 0.5 * ((1 / (P(eps1) * P(eps2))) * (Vnext(gamma) - (P(eps1) + P(eps2)) * zmp - (1 - (P(eps1) * cosf(xmp))) * ymp));
    }
    
}

#undef name