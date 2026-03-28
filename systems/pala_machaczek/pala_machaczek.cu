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

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters, PerThread* pt)
{
    const numb x = V(x);
    const numb y = V(y);
    const numb z = V(z);

    const numb R = P(R);
    const numb L = P(L);
    const numb C = P(C);
    const numb m = P(m);

    const numb h = H;
    const numb h2 = h * (numb)0.5;

    const numb invL = (numb)1.0 / L;
    const numb invRC = (numb)1.0 / (R * C);
    const numb Rp1 = R + (numb)1.0;

    numb sx, sy, sz;
    numb xmp, ymp, zmp;
    numb kx, ky, kz;

    // k1
    {
        kx = invL * (y - x * exp(m * log(z)));
        ky = invRC * (Rp1 - y - R * x);
        kz = x * x - z;

        sx = kx;
        sy = ky;
        sz = kz;

        xmp = x + h2 * kx;
        ymp = y + h2 * ky;
        zmp = z + h2 * kz;
    }
    // k2
    {
        kx = invL * (ymp - xmp * exp(m * log(zmp)));
        ky = invRC * (Rp1 - ymp - R * xmp);
        kz = xmp * xmp - zmp;

        sx += (numb)2.0 * kx;
        sy += (numb)2.0 * ky;
        sz += (numb)2.0 * kz;

        xmp = x + h2 * kx;
        ymp = y + h2 * ky;
        zmp = z + h2 * kz;
    }
    // k3
    {
        kx = invL * (ymp - xmp * exp(m * log(zmp)));
        ky = invRC * (Rp1 - ymp - R * xmp);
        kz = xmp * xmp - zmp;

        sx += (numb)2.0 * kx;
        sy += (numb)2.0 * ky;
        sz += (numb)2.0 * kz;

        xmp = x + h * kx;
        ymp = y + h * ky;
        zmp = z + h * kz;
    }
    // k4
    {
        kx = invL * (ymp - xmp * exp(m * log(zmp)));
        ky = invRC * (Rp1 - ymp - R * xmp);
        kz = xmp * xmp - zmp;

        sx += kx;
        sy += ky;
        sz += kz;
    }

    const numb h6 = h / (numb)6.0;

    Vnext(x) = x + h6 * sx;
    Vnext(y) = y + h6 * sy;
    Vnext(z) = z + h6 * sz;
}


#undef name
