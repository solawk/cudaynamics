#include "pala_machaczek.h"

#define name pala_machaczek

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { R, L, C, m, COUNT };
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
    const numb x = V(x);
    const numb y = V(y);
    const numb z = V(z);

    const numb R = P(R);
    const numb L = P(L);
    const numb C = P(C);
    const numb m = P(m);

    const numb h1 = 0.5f * H;
    const numb h2 = H;

    numb kx1 = (1.0f / L) * (y - x * powf(z, m));
    numb ky1 = (1.0f / (R * C)) * (R + 1.0f - y - R * x);
    numb kz1 = x * x - z;

    numb xmp = x + h1 * kx1;
    numb ymp = y + h1 * ky1;
    numb zmp = z + h1 * kz1;

    numb kx2 = (1.0f / L) * (ymp - xmp * powf(zmp, m));
    numb ky2 = (1.0f / (R * C)) * (R + 1.0f - ymp - R * xmp);
    numb kz2 = xmp * xmp - zmp;

    xmp = x + h1 * kx2;
    ymp = y + h1 * ky2;
    zmp = z + h1 * kz2;

    numb kx3 = (1.0f / L) * (ymp - xmp * powf(zmp, m));
    numb ky3 = (1.0f / (R * C)) * (R + 1.0f - ymp - R * xmp);
    numb kz3 = xmp * xmp - zmp;

    xmp = x + h2 * kx3;
    ymp = y + h2 * ky3;
    zmp = z + h2 * kz3;

    numb kx4 = (1.0f / L) * (ymp - xmp * powf(zmp, m));
    numb ky4 = (1.0f / (R * C)) * (R + 1.0f - ymp - R * xmp);
    numb kz4 = xmp * xmp - zmp;

    Vnext(x) = x + h2 * (kx1 + 2.0f * kx2 + 2.0f * kx3 + kx4) / 6.0f;
    Vnext(y) = y + h2 * (ky1 + 2.0f * ky2 + 2.0f * ky3 + ky4) / 6.0f;
    Vnext(z) = z + h2 * (kz1 + 2.0f * kz2 + 2.0f * kz3 + kz4) / 6.0f;
}

#undef name