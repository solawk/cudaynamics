#include "hindmarsh_rose.h"

#define name hindmarsh_rose

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, d, r, s, e, Iext, symmetry, method, COUNT };
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
        Vnext(x) = V(x) + H * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + P(Iext));
        Vnext(y) = V(y) + H * (P(c) - P(d) * V(x) * V(x) - V(y));
        Vnext(z) = V(z) + H * (P(r) * (P(s) * (V(x) + P(e)) - V(z)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * 0.5 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + P(Iext));
        numb ymp = V(y) + H * 0.5 * (P(c) - P(d) * V(x) * V(x) - V(y));
        numb zmp = V(z) + H * 0.5 * (P(r) * (P(s) * (V(x) + P(e)) - V(z)));

        Vnext(x) = V(x) + H * (ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext));
        Vnext(y) = V(y) + H * (P(c) - P(d) * xmp * xmp - ymp);
        Vnext(z) = V(z) + H * (P(r) * (P(s) * (xmp + P(e)) - zmp));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + P(Iext);
        numb ky1 = P(c) - P(d) * V(x) * V(x) - V(y);
        numb kz1 = P(r) * (P(s) * (V(x) + P(e)) - V(z));

        numb xmp = V(x) + 0.5 * H * kx1;
        numb ymp = V(y) + 0.5 * H * ky1;
        numb zmp = V(z) + 0.5 * H * kz1;

        numb kx2 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext);
        numb ky2 = P(c) - P(d) * xmp * xmp - ymp;
        numb kz2 = P(r) * (P(s) * (xmp + P(e)) - zmp);

        xmp = V(x) + 0.5 * H * kx2;
        ymp = V(y) + 0.5 * H * ky2;
        zmp = V(z) + 0.5 * H * kz2;

        numb kx3 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext);
        numb ky3 = P(c) - P(d) * xmp * xmp - ymp;
        numb kz3 = P(r) * (P(s) * (xmp + P(e)) - zmp);

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext);
        numb ky4 = P(c) - P(d) * xmp * xmp - ymp;
        numb kz4 = P(r) * (P(s) * (xmp + P(e)) - zmp);

        Vnext(x) = V(x) + H * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4) / 6.0;
        Vnext(y) = V(y) + H * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4) / 6.0;
        Vnext(z) = V(z) + H * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4) / 6.0;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + P(Iext));
        numb ymp = V(y) + h1 * (P(c) - P(d) * xmp * xmp - V(y));
        numb zmp = V(z) + h1 * (P(r) * (P(s) * (xmp + P(e)) - V(z)));

        Vnext(z) = (zmp + P(r) * P(s) * (xmp + P(e)) * h2) / (1 + P(r) * h2);
        Vnext(y) = (ymp + (P(c) - P(d) * xmp * xmp) * h2) / (1 + h2);

        Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - Vnext(z) + P(Iext));
        Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * Vnext(x) * Vnext(x) * Vnext(x) + P(b) * Vnext(x) * Vnext(x) - Vnext(z) + P(Iext));
    }
}

#undef name