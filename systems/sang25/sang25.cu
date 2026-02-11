#include "sang25.h"

#define name sang25

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, mu, omega, symmetry, method, COUNT };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
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
        Vnext(x) = V(x) + H * (-V(y));
        Vnext(y) = V(y) + H * (V(x) + P(c) * V(y) + P(a) * V(z));
        Vnext(z) = V(z) + H * (-P(mu) * V(z) + P(b) * cos(P(omega) * V(y)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * (numb)0.5 * (-V(y));
        numb ymp = V(y) + H * (numb)0.5 * (V(x) + P(c) * V(y) + P(a) * V(z));
        numb zmp = V(z) + H * (numb)0.5 * (-P(mu) * V(z) + P(b) * cos(P(omega) * V(y)));

        Vnext(x) = V(x) + H * (-ymp);
        Vnext(y) = V(y) + H * (xmp + P(c) * ymp + P(a) * zmp);
        Vnext(z) = V(z) + H * (-P(mu) * zmp + P(b) * cos(P(omega) * ymp));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = -V(y);
        numb ky1 = V(x) + P(c) * V(y) + P(a) * V(z);
        numb kz1 = -P(mu) * V(z) + P(b) * cos(P(omega) * V(y));

        numb xmp = V(x) + (numb)0.5 * H * kx1;
        numb ymp = V(y) + (numb)0.5 * H * ky1;
        numb zmp = V(z) + (numb)0.5 * H * kz1;

        numb kx2 = -ymp;
        numb ky2 = xmp + P(c) * ymp + P(a) * zmp;
        numb kz2 = -P(mu) * zmp + P(b) * cos(P(omega) * ymp);

        xmp = V(x) + (numb)0.5 * H * kx2;
        ymp = V(y) + (numb)0.5 * H * ky2;
        zmp = V(z) + (numb)0.5 * H * kz2;

        numb kx3 = -ymp;
        numb ky3 = xmp + P(c) * ymp + P(a) * zmp;
        numb kz3 = -P(mu) * zmp + P(b) * cos(P(omega) * ymp);

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = -ymp;
        numb ky4 = xmp + P(c) * ymp + P(a) * zmp;
        numb kz4 = -P(mu) * zmp + P(b) * cos(P(omega) * ymp);

        Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
        Vnext(y) = V(y) + H * (ky1 + (numb)2.0 * ky2 + (numb)2.0 * ky3 + ky4) / (numb)6.0;
        Vnext(z) = V(z) + H * (kz1 + (numb)2.0 * kz2 + (numb)2.0 * kz3 + kz4) / (numb)6.0;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = (numb)0.5 * H - P(symmetry);
        numb h2 = (numb)0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (-V(y));
        numb ymp = V(y) + h1 * (xmp + P(c) * V(y) + P(a) * V(z));
        numb zmp = V(z) + h1 * (-P(mu) * V(z) + P(b) * cos(P(omega) * ymp));

        Vnext(z) = (zmp + P(b) * cos(P(omega) * ymp) * h2) / ((numb)1.0 + P(mu) * h2);
        Vnext(y) = (ymp + (xmp + P(a) * Vnext(z)) * h2) / ((numb)1.0 - P(c) * h2);
        Vnext(x) = xmp + h2 * (-Vnext(y));
    }
}

#undef name