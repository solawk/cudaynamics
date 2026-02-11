#include "sang26.h"

#define name sang26

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
        Vnext(z) = V(z) + H * (-V(z) + sin(P(b) * V(x)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * (numb)0.5 * (-V(y) + P(a) * V(z));
        numb ymp = V(y) + H * (numb)0.5 * V(x);
        numb zmp = V(z) + H * (numb)0.5 * (-V(z) + sin(P(b) * V(x)));

        Vnext(x) = V(x) + H * (-ymp + P(a) * zmp);
        Vnext(y) = V(y) + H * xmp;
        Vnext(z) = V(z) + H * (-zmp + sin(P(b) * xmp));
    }

    /*ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(sigma) * (V(y) - V(x));
        numb ky1 = V(x) * (P(rho) - V(z)) - V(y);
        numb kz1 = V(x) * V(y) - P(beta) * V(z);

        numb xmp = V(x) + (numb)0.5 * H * kx1;
        numb ymp = V(y) + (numb)0.5 * H * ky1;
        numb zmp = V(z) + (numb)0.5 * H * kz1;

        numb kx2 = P(sigma) * (ymp - xmp);
        numb ky2 = xmp * (P(rho) - zmp) - ymp;
        numb kz2 = xmp * ymp - P(beta) * zmp;

        xmp = V(x) + (numb)0.5 * H * kx2;
        ymp = V(y) + (numb)0.5 * H * ky2;
        zmp = V(z) + (numb)0.5 * H * kz2;

        numb kx3 = P(sigma) * (ymp - xmp);
        numb ky3 = xmp * (P(rho) - zmp) - ymp;
        numb kz3 = xmp * ymp - P(beta) * zmp;

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = P(sigma) * (ymp - xmp);
        numb ky4 = xmp * (P(rho) - zmp) - ymp;
        numb kz4 = xmp * ymp - P(beta) * zmp;

        Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
        Vnext(y) = V(y) + H * (ky1 + (numb)2.0 * ky2 + (numb)2.0 * ky3 + ky4) / (numb)6.0;
        Vnext(z) = V(z) + H * (kz1 + (numb)2.0 * kz2 + (numb)2.0 * kz3 + kz4) / (numb)6.0;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = (numb)0.5 * H - P(symmetry);
        numb h2 = (numb)0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (P(sigma) * (V(y) - V(x)));
        numb ymp = V(y) + h1 * (xmp * (P(rho) - V(z)) - V(y));
        numb zmp = V(z) + h1 * (xmp * ymp - P(beta) * V(z));

        Vnext(z) = (zmp + xmp * ymp * h2) / ((numb)1.0 + P(beta) * h2);
        Vnext(y) = (ymp + xmp * (P(rho) - Vnext(z)) * h2) / ((numb)1.0 + h2);
        Vnext(x) = (xmp + P(sigma) * Vnext(y) * h2) / ((numb)1.0 + P(sigma) * h2);
    }*/
}

#undef name