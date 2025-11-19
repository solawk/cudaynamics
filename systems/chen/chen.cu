#include "main.h"
#include "chen.h"

#define name chen

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { alpha, beta, gamma, delta, symmetry, method, COUNT };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
}

__global__ void kernelProgram_(name)(Computation* data)
{
    uint64_t variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
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

__device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters)
{

    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + H * (P(alpha) * V(x) - V(y) * V(z));
        Vnext(y) = V(y) + H * (P(beta) * V(y) + V(x) * V(z));
        Vnext(z) = V(z) + H * (P(delta) * V(z) + V(x) * V(y) / P(gamma));
    }


    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + 0.5 * H * (P(alpha) * V(x) - V(y) * V(z));
        numb ymp = V(y) + 0.5 * H * (P(beta) * V(y) + V(x) * V(z));
        numb zmp = V(z) + 0.5 * H * (P(delta) * V(z) + V(x) * V(y) / P(gamma));

        Vnext(x) = V(x) + H * (P(alpha) * xmp - ymp * zmp);
        Vnext(y) = V(y) + H * (P(beta) * ymp + xmp * zmp);
        Vnext(z) = V(z) + H * (P(delta) * zmp + xmp * ymp / P(gamma));
    }


    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(alpha) * V(x) - V(y) * V(z);
        numb ky1 = P(beta) * V(y) + V(x) * V(z);
        numb kz1 = P(delta) * V(z) + V(x) * V(y) / P(gamma);

        numb xmp = V(x) + 0.5 * H * kx1;
        numb ymp = V(y) + 0.5 * H * ky1;
        numb zmp = V(z) + 0.5 * H * kz1;

        numb kx2 = P(alpha) * xmp - ymp * zmp;
        numb ky2 = P(beta) * ymp + xmp * zmp;
        numb kz2 = P(delta) * zmp + xmp * ymp / P(gamma);

        xmp = V(x) + 0.5 * H * kx2;
        ymp = V(y) + 0.5 * H * ky2;
        zmp = V(z) + 0.5 * H * kz2;

        numb kx3 = P(alpha) * xmp - ymp * zmp;
        numb ky3 = P(beta) * ymp + xmp * zmp;
        numb kz3 = P(delta) * zmp + xmp * ymp / P(gamma);

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = P(alpha) * xmp - ymp * zmp;
        numb ky4 = P(beta) * ymp + xmp * zmp;
        numb kz4 = P(delta) * zmp + xmp * ymp / P(gamma);

        Vnext(x) = V(x) + H * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + H * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + H * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }


    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (P(alpha) * V(x) - V(y) * V(z));
        numb ymp = V(y) + h1 * (P(beta) * V(y) + xmp * V(z));
        numb zmp = V(z) + h1 * (P(delta) * V(z) + xmp * ymp / P(gamma));

        Vnext(z) = (zmp + xmp * ymp / P(gamma) * h2) / (1 - P(delta) * h2);
        Vnext(y) = (ymp + xmp * Vnext(z) * h2) / (1 - P(beta) * h2);
        Vnext(x) = (xmp - Vnext(y) * Vnext(z) * h2) / (1 - P(alpha) * h2);      
    }

}

#undef name