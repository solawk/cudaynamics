#include "dadras_momeni.h"

#define name dadras_momeni

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, d, e, symmetry, method, COUNT };
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
        Vnext(x) = V(x) + H * (V(y) + P(a) * V(x) + P(b) * V(y) * V(z));
        Vnext(y) = V(y) + H * (P(c) * V(y) - V(x) * V(z) + V(z));
        Vnext(z) = V(z) + H * (P(d) * V(x) * V(y) - P(e) * V(z));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * 0.5 * (V(y) - P(a) * V(x) + P(b) * V(y) * V(z));
        numb ymp = V(y) + H * 0.5 * (P(c) * V(y) - V(x) * V(z) + V(z));
        numb zmp = V(z) + H * 0.5 * (P(d) * V(x) * V(y) - P(e) * V(z));

        Vnext(x) = V(x) + H * (ymp - P(a) * xmp + P(b) * ymp * zmp);
        Vnext(y) = V(y) + H * (P(c) * ymp - xmp * zmp + zmp);
        Vnext(z) = V(z) + H * (P(d) * xmp * ymp - P(e) * zmp);
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = V(y) - P(a) * V(x) + P(b) * V(y) * V(z);
        numb ky1 = P(c) * V(y) - V(x) * V(z) + V(z);
        numb kz1 = P(d) * V(x) * V(y) - P(e) * V(z);

        numb xmp = V(x) + 0.5 * H * kx1;
        numb ymp = V(y) + 0.5 * H * ky1;
        numb zmp = V(z) + 0.5 * H * kz1;

        numb kx2 = ymp - P(a) * xmp + P(b) * ymp * zmp;
        numb ky2 = P(c) * ymp - xmp * zmp + zmp;
        numb kz2 = P(d) * xmp * ymp - P(e) * zmp;

        xmp = V(x) + 0.5 * H * kx2;
        ymp = V(y) + 0.5 * H * ky2;
        zmp = V(z) + 0.5 * H * kz2;

        numb kx3 = ymp - P(a) * xmp + P(b) * ymp * zmp;
        numb ky3 = P(c) * ymp - xmp * zmp + zmp;
        numb kz3 = P(d) * xmp * ymp - P(e) * zmp;

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = ymp - P(a) * xmp + P(b) * ymp * zmp;
        numb ky4 = P(c) * ymp - xmp * zmp + zmp;
        numb kz4 = P(d) * xmp * ymp - P(e) * zmp;

        Vnext(x) = V(x) + H * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + H * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + H * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (V(y) - P(a) * V(x) + P(b) * V(y) * V(z));
        numb ymp = V(y) + h1 * (P(c) * V(y) - xmp * V(z) + V(z));
        numb zmp = V(z) + h1 * (P(d) * xmp * ymp - P(e) * V(z));
        
        Vnext(z) = (zmp + P(d) * xmp * ymp * h2) / (1 + P(e) * h2);
        Vnext(y) = (ymp - xmp * Vnext(z) * h2 + Vnext(z) * h2) / (1 - h2 * P(c));
        Vnext(x) = (xmp + h2 * Vnext(y) + P(b) * Vnext(y) * Vnext(z) * h2) / (1 + P(a) * h2);
    }
}

#undef name