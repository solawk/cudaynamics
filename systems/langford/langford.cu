#include "langford.h"

#define name langford

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, d, e, f, g, symmetry, method, COUNT };
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
        Vnext(x) = V(x) + H * ((V(z) - P(b)) * V(x) - P(d) * V(y));
        Vnext(y) = V(y) + H * (P(d) * V(x) + (V(z) - P(b)) * V(y));
        Vnext(z) = V(z) + H * (P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (V(x) * V(x) + V(y) * V(y)) * ((numb)1.0 + P(e) * V(z)) + P(f) * V(z));
    }


    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + (numb)0.5 * H * ((V(z) - P(b)) * V(x) - P(d) * V(y));
        numb ymp = V(y) + (numb)0.5 * H * (P(d) * V(x) + (V(z) - P(b)) * V(y));
        numb zmp = V(z) + (numb)0.5 * H * (P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (V(x) * V(x) + V(y) * V(y)) * ((numb)1.0 + P(e) * V(z)) + P(f) * V(z));

        Vnext(x) = V(x) + H * ((zmp - P(b)) * xmp - P(d) * ymp);
        Vnext(y) = V(y) + H * (P(d) * xmp + (zmp - P(b)) * ymp);
        Vnext(z) = V(z) + H * (P(c) + P(a) * zmp - zmp * zmp * zmp / P(g) - (xmp * xmp + ymp * ymp) * ((numb)1.0 + P(e) * zmp) + P(f) * zmp);
    }


    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = (V(z) - P(b)) * V(x) - P(d) * V(y);
        numb ky1 = P(d) * V(x) + (V(z) - P(b)) * V(y);
        numb kz1 = P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (V(x) * V(x) + V(y) * V(y)) * ((numb)1.0 + P(e) * V(z)) + P(f) * V(z);

        numb xmp = V(x) + (numb)0.5 * H * kx1;
        numb ymp = V(y) + (numb)0.5 * H * ky1;
        numb zmp = V(z) + (numb)0.5 * H * kz1;

        numb kx2 = (zmp - P(b)) * xmp - P(d) * ymp;
        numb ky2 = P(d) * xmp + (zmp - P(b)) * ymp;
        numb kz2 = P(c) + P(a) * zmp - zmp * zmp * zmp / P(g) - (xmp * xmp + ymp * ymp) * ((numb)1.0 + P(e) * zmp) + P(f) * zmp;

        xmp = V(x) + (numb)0.5 * H * kx2;
        ymp = V(y) + (numb)0.5 * H * ky2;
        zmp = V(z) + (numb)0.5 * H * kz2;

        numb kx3 = (zmp - P(b)) * xmp - P(d) * ymp;
        numb ky3 = P(d) * xmp + (zmp - P(b)) * ymp;
        numb kz3 = P(c) + P(a) * zmp - zmp * zmp * zmp / P(g) - (xmp * xmp + ymp * ymp) * ((numb)1.0 + P(e) * zmp) + P(f) * zmp;

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = (zmp - P(b)) * xmp - P(d) * ymp;
        numb ky4 = P(d) * xmp + (zmp - P(b)) * ymp;
        numb kz4 = P(c) + P(a) * zmp - zmp * zmp * zmp / P(g) - (xmp * xmp + ymp * ymp) * ((numb)1.0 + P(e) * zmp) + P(f) * zmp;

        Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
        Vnext(y) = V(y) + H * (ky1 + (numb)2.0 * ky2 + (numb)2.0 * ky3 + ky4) / (numb)6.0;
        Vnext(z) = V(z) + H * (kz1 + (numb)2.0 * kz2 + (numb)2.0 * kz3 + kz4) / (numb)6.0;
    }


    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = (numb)0.5 * H - P(symmetry);
        numb h2 = (numb)0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * ((V(z) - P(b)) * V(x) - P(d) * V(y));
        numb ymp = V(y) + h1 * (P(d) * xmp + (V(z) - P(b)) * V(y));
        numb zmp = V(z) + h1 * (P(c) + P(a) * V(z) - V(z) * V(z) * V(z) / P(g) - (xmp * xmp + ymp * ymp) * ((numb)1.0 + P(e) * V(z)) + P(f) * V(z));

        Vnext(z) = zmp + h2 * (P(c) + P(a) * zmp - zmp * zmp * zmp / P(g) - (xmp * xmp + ymp * ymp) * ((numb)1.0 + P(e) * zmp) + P(f) * zmp);
        Vnext(z) = zmp + h2 * (P(c) + P(a) * Vnext(z) - Vnext(z) * Vnext(z) * Vnext(z) / P(g) - (xmp * xmp + ymp * ymp) * ((numb)1.0 + P(e) * Vnext(z)) + P(f) * Vnext(z));
        Vnext(y) = (ymp + h2 * P(d) * xmp) / ((numb)1.0 - h2 * (Vnext(z) - P(b)));
        Vnext(x) = (xmp - h2 * P(d) * Vnext(y)) / ((numb)1.0 - h2 * (Vnext(z) - P(b)));
    }
}

#undef name