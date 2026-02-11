#include "jj_rcls.h"

#define name jj_rcls

namespace attributes
{
    enum variables { x1, sin_x1, x2, x3 };
    enum parameters { betaL, betaC, i, gThreshold, Rn, Rsg, symmetry, method, COUNT };
    enum methods { ExplicitEuler, SemiExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
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
        Vnext(x1) = fmod(V(x1) + H * V(x2), (numb)2.0 * (numb)3.141592);
        Vnext(sin_x1) = sin(Vnext(x1));
        Vnext(x2) = V(x2) + H * ((numb)1.0 / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sin(V(x1)) - V(x3));
        Vnext(x3) = V(x3) + H * ((numb)1.0 / P(betaL)) * (V(x2) - V(x3));
    }

    ifMETHOD(P(method), SemiExplicitEuler)
    {
        Vnext(x1) = fmod(V(x1) + H * V(x2), (numb)2.0 * (numb)3.141592);
        Vnext(sin_x1) = sin(Vnext(x1));
        Vnext(x3) = V(x3) + H * ((numb)1.0 / P(betaL)) * (V(x2) - V(x3));
        Vnext(x2) = V(x2) + H * ((numb)1.0 / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sin(Vnext(x1)) - Vnext(x3));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb x1mp = fmod(V(x1) + H * (numb)0.5 * V(x2), (numb)2.0 * (numb)3.141592);
        numb x2mp = V(x2) + H * (numb)0.5 * ((numb)1.0 / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sin(V(x1)) - V(x3));
        numb x3mp = V(x3) + H * (numb)0.5 * ((numb)1.0 / P(betaL)) * (V(x2) - V(x3));

        Vnext(x1) = fmod(V(x1) + H * x2mp, (numb)2.0 * (numb)3.141592);
        Vnext(sin_x1) = sin(Vnext(x1));
        Vnext(x2) = V(x2) + H * ((numb)1.0 / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sin(x1mp) - x3mp);
        Vnext(x3) = V(x3) + H * ((numb)1.0 / P(betaL)) * (x2mp - x3mp);
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx11 = V(x2);
        numb kx21 = ((numb)1.0 / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sin(V(x1)) - V(x3));
        numb kx31 = ((numb)1.0 / P(betaL)) * (V(x2) - V(x3));

        numb x1mp = fmod(V(x1) + H * (numb)0.5 * kx11, (numb)2.0 * (numb)3.141592);
        numb x2mp = V(x2) + H * (numb)0.5 * kx21;
        numb x3mp = V(x3) + H * (numb)0.5 * kx31;

        numb kx12 = x2mp;
        numb kx22 = ((numb)1.0 / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sin(x1mp) - x3mp);
        numb kx32 = ((numb)1.0 / P(betaL)) * (x2mp - x3mp);

        x1mp = fmod(V(x1) + H * (numb)0.5 * kx12, (numb)2.0 * (numb)3.141592);
        x2mp = V(x2) + H * (numb)0.5 * kx22;
        x3mp = V(x3) + H * (numb)0.5 * kx32;

        numb kx13 = x2mp;
        numb kx23 = ((numb)1.0 / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sin(x1mp) - x3mp);
        numb kx33 = ((numb)1.0 / P(betaL)) * (x2mp - x3mp);

        x1mp = fmod(V(x1) + H * kx13, (numb)2.0 * (numb)3.141592);
        x2mp = V(x2) + H * kx23;
        x3mp = V(x3) + H * kx33;

        numb kx14 = x2mp;
        numb kx24 = ((numb)1.0 / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sin(x1mp) - x3mp);
        numb kx34 = ((numb)1.0 / P(betaL)) * (x2mp - x3mp);

        Vnext(x1) = fmod(V(x1) + H * (kx11 + (numb)2.0 * kx12 + (numb)2.0 * kx13 + kx14) / (numb)6.0, (numb)2.0 * (numb)3.141592);
        Vnext(sin_x1) = sin(Vnext(x1));
        Vnext(x2) = V(x2) + H * (kx21 + (numb)2.0 * kx22 + (numb)2.0 * kx23 + kx24) / (numb)6.0;
        Vnext(x3) = V(x3) + H * (kx31 + (numb)2.0 * kx32 + (numb)2.0 * kx33 + kx34) / (numb)6.0;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = (numb)0.5 * H - P(symmetry);
        numb h2 = (numb)0.5 * H + P(symmetry);

        numb x1mp = fmod(V(x1) + h1 * V(x2), (numb)2.0 * (numb)3.141592);
        numb x3mp = V(x3) + h1 * ((numb)1.0 / P(betaL)) * (V(x2) - V(x3));
        numb x2mp = V(x2) + h1 * ((numb)1.0 / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sin(x1mp) - x3mp);

        Vnext(x2) = x2mp + h2 * ((numb)1.0 / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sin(x1mp) - x3mp);
        Vnext(x2) = x2mp + h2 * ((numb)1.0 / P(betaC)) * (P(i) - ((Vnext(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * Vnext(x2) - sin(x1mp) - x3mp);
        Vnext(x3) = (x3mp + h2 * ((numb)1.0 / P(betaL)) * Vnext(x2)) / ((numb)1.0 + h2 * ((numb)1.0 / P(betaL)));
        Vnext(x1) = fmod(x1mp + h2 * Vnext(x2), (numb)2.0 * (numb)3.141592);
        Vnext(sin_x1) = sin(Vnext(x1));
    }
}

#undef name