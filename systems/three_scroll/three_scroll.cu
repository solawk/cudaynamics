#include "three_scroll.h"

#define name three_scroll

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, d, e, f, symmetry, method, COUNT };
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
        Vnext(x) = V(x) + H * (P(a) * (V(y) - V(x)) + P(d) * V(x) * V(z));
        Vnext(y) = V(y) + H * (V(x) * (P(b) - V(z)) + P(f) * V(y));
        Vnext(z) = V(z) + H * (P(c) * V(z) + V(x) * (V(y) - P(e) * V(x)));
    }


    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + (numb)0.5 * H * (P(a) * (V(y) - V(x)) + P(d) * V(x) * V(z));
        numb ymp = V(y) + (numb)0.5 * H * (V(x) * (P(b) - V(z)) + P(f) * V(y));
        numb zmp = V(z) + (numb)0.5 * H * (P(c) * V(z) + V(x) * (V(y) - P(e) * V(x)));

        Vnext(x) = V(x) + H * (P(a) * (ymp - xmp) + P(d) * xmp * zmp);
        Vnext(y) = V(y) + H * (xmp * (P(b) - zmp) + P(f) * ymp);
        Vnext(z) = V(z) + H * (P(c) * zmp + xmp * (ymp - P(e) * xmp));
    }


    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(a) * (V(y) - V(x)) + P(d) * V(x) * V(z);
        numb ky1 = V(x) * (P(b) - V(z)) + P(f) * V(y);
        numb kz1 = P(c) * V(z) + V(x) * (V(y) - P(e) * V(x));

        numb xmp = V(x) + (numb)0.5 * H * kx1;
        numb ymp = V(y) + (numb)0.5 * H * ky1;
        numb zmp = V(z) + (numb)0.5 * H * kz1;

        numb kx2 = P(a) * (ymp - xmp) + P(d) * xmp * zmp;
        numb ky2 = xmp * (P(b) - zmp) + P(f) * ymp;
        numb kz2 = P(c) * zmp + xmp * (ymp - P(e) * xmp);

        xmp = V(x) + (numb)0.5 * H * kx2;
        ymp = V(y) + (numb)0.5 * H * ky2;
        zmp = V(z) + (numb)0.5 * H * kz2;

        numb kx3 = P(a) * (ymp - xmp) + P(d) * xmp * zmp;
        numb ky3 = xmp * (P(b) - zmp) + P(f) * ymp;
        numb kz3 = P(c) * zmp + xmp * (ymp - P(e) * xmp);

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = P(a) * (ymp - xmp) + P(d) * xmp * zmp;
        numb ky4 = xmp * (P(b) - zmp) + P(f) * ymp;
        numb kz4 = P(c) * zmp + xmp * (ymp - P(e) * xmp);

        Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
        Vnext(y) = V(y) + H * (ky1 + (numb)2.0 * ky2 + (numb)2.0 * ky3 + ky4) / (numb)6.0;
        Vnext(z) = V(z) + H * (kz1 + (numb)2.0 * kz2 + (numb)2.0 * kz3 + kz4) / (numb)6.0;
    }


    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = (numb)0.5 * H - P(symmetry);
        numb h2 = (numb)0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (P(a) * (V(y) - V(x)) + P(d) * V(x) * V(z));
        numb ymp = V(y) + h1 * (xmp * (P(b) - V(z)) + P(f) * V(y));
        numb zmp = V(z) + h1 * (P(c) * V(z) + xmp * (ymp - P(e) * xmp));

        Vnext(z) = (zmp + h2 * xmp * (ymp - P(e) * xmp)) / ((numb)1.0 - h2 * P(c));
        Vnext(y) = (ymp + h2 * xmp * (P(b) - Vnext(z))) / ((numb)1.0 - h2 * P(f));
        Vnext(x) = (xmp + h2 * P(a) * Vnext(y)) / ((numb)1.0 + h2 * (P(a) - P(d) * Vnext(z)));
    }
}

#undef name