#include "b.h"

#define name b

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, symmetry, method, COUNT };
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
    /*
    *   dx = -y + a * z;
        dy= x;
        dz = -z + sin(b * x);
    */

    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + H * (-V(y) + P(a) * V(z));
        Vnext(y) = V(y) + H * V(x);
        Vnext(z) = V(z) + H * (-V(z) + sinf(P(b) * V(x)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * 0.5f * (-V(y) + P(a) * V(z));
        numb ymp = V(y) + H * 0.5f * V(x);
        numb zmp = V(z) + H * 0.5f * (-V(z) + sinf(P(b) * V(x)));

        Vnext(x) = V(x) + H * (-ymp + P(a) * zmp);
        Vnext(y) = V(y) + H * xmp;
        Vnext(z) = V(z) + H * (-zmp + sinf(P(b) * xmp));
    }

    /*ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(sigma) * (V(y) - V(x));
        numb ky1 = V(x) * (P(rho) - V(z)) - V(y);
        numb kz1 = V(x) * V(y) - P(beta) * V(z);

        numb xmp = V(x) + 0.5f * H * kx1;
        numb ymp = V(y) + 0.5f * H * ky1;
        numb zmp = V(z) + 0.5f * H * kz1;

        numb kx2 = P(sigma) * (ymp - xmp);
        numb ky2 = xmp * (P(rho) - zmp) - ymp;
        numb kz2 = xmp * ymp - P(beta) * zmp;

        xmp = V(x) + 0.5f * H * kx2;
        ymp = V(y) + 0.5f * H * ky2;
        zmp = V(z) + 0.5f * H * kz2;

        numb kx3 = P(sigma) * (ymp - xmp);
        numb ky3 = xmp * (P(rho) - zmp) - ymp;
        numb kz3 = xmp * ymp - P(beta) * zmp;

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = P(sigma) * (ymp - xmp);
        numb ky4 = xmp * (P(rho) - zmp) - ymp;
        numb kz4 = xmp * ymp - P(beta) * zmp;

        Vnext(x) = V(x) + H * (kx1 + 2.0f * kx2 + 2.0f * kx3 + kx4) / 6.0f;
        Vnext(y) = V(y) + H * (ky1 + 2.0f * ky2 + 2.0f * ky3 + ky4) / 6.0f;
        Vnext(z) = V(z) + H * (kz1 + 2.0f * kz2 + 2.0f * kz3 + kz4) / 6.0f;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5f * H - P(symmetry);
        numb h2 = 0.5f * H + P(symmetry);

        numb xmp = V(x) + h1 * (P(sigma) * (V(y) - V(x)));
        numb ymp = V(y) + h1 * (xmp * (P(rho) - V(z)) - V(y));
        numb zmp = V(z) + h1 * (xmp * ymp - P(beta) * V(z));

        Vnext(z) = (zmp + xmp * ymp * h2) / (1.0f + P(beta) * h2);
        Vnext(y) = (ymp + xmp * (P(rho) - Vnext(z)) * h2) / (1.0f + h2);
        Vnext(x) = (xmp + P(sigma) * Vnext(y) * h2) / (1.0f + P(sigma) * h2);
    }*/
}

#undef name