#include "rossler.h"

#define name rossler

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, symmetry, method, COUNT };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
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
        Vnext(x) = V(x) + H * (-(V(y) + V(z)));
        Vnext(y) = V(y) + H * (V(x) + P(a) * V(y));
        Vnext(z) = V(z) + H * (P(b) + V(z) * (V(x) - P(c)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * 0.5f * (-(V(y) + V(z)));
        numb ymp = V(y) + H * 0.5f * (V(x) + P(a) * V(y));
        numb zmp = V(z) + H * 0.5f * (P(b) + V(z) * (V(x) - P(c)));

        Vnext(x) = V(x) + H * (-(ymp + zmp));
        Vnext(y) = V(y) + H * (xmp + P(a) * ymp);
        Vnext(z) = V(z) + H * (P(b) + zmp * (xmp - P(c)));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = -(V(y) + V(z));
        numb ky1 = V(x) + P(a) * V(y);
        numb kz1 = P(b) + V(z) * (V(x) - P(c));

        numb xmp = V(x) + 0.5f * H * kx1;
        numb ymp = V(y) + 0.5f * H * ky1;
        numb zmp = V(z) + 0.5f * H * kz1;

        numb kx2 = -(ymp + zmp);
        numb ky2 = xmp + P(a) * ymp;
        numb kz2 = P(b) + zmp * (xmp - P(c));

        xmp = V(x) + 0.5f * H * kx2;
        ymp = V(y) + 0.5f * H * ky2;
        zmp = V(z) + 0.5f * H * kz2;

        numb kx3 = -(ymp + zmp);
        numb ky3 = xmp + P(a) * ymp;
        numb kz3 = P(b) + zmp * (xmp - P(c));

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = -(ymp + zmp);
        numb ky4 = xmp + P(a) * ymp;
        numb kz4 = P(b) + zmp * (xmp - P(c));

        Vnext(x) = V(x) + H * (kx1 + 2.0f * kx2 + 2.0f * kx3 + kx4) / 6.0f;
        Vnext(y) = V(y) + H * (ky1 + 2.0f * ky2 + 2.0f * ky3 + ky4) / 6.0f;
        Vnext(z) = V(z) + H * (kz1 + 2.0f * kz2 + 2.0f * kz3 + kz4) / 6.0f;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5f * H - P(symmetry);
        numb h2 = 0.5f * H + P(symmetry);

        numb zmp = V(z) + h1 * (P(b) + V(z) * (V(x) - P(c)));
        numb ymp = V(y) + h1 * (V(x) + P(a) * V(y));
        numb xmp = V(x) + h1 * (-(ymp + zmp));
        
        Vnext(x) = xmp + h2 * (-(ymp + zmp));
        Vnext(y) = (ymp + h2 * Vnext(x)) / (1 - P(a) * h2);
        Vnext(z) = (zmp + h2 * P(b)) / (1 - (Vnext(x) - P(c)) * h2);
    }
}

#undef name