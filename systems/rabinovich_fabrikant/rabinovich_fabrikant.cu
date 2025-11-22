#include "rabinovich_fabrikant.h"

#define name rabinovich_fabrikant

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { alpha, betta, gamma, delta, epsilon, symmetry, method, COUNT };
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
        Vnext(x) = V(x) + H * (V(y) * (V(z) - P(betta) + pow(V(x), 2)) + P(gamma) * V(x));
        Vnext(y) = V(y) + H * (V(x) * (P(delta) * V(z) + P(betta) - pow(V(x), 2)) + P(gamma) * V(y));
        Vnext(z) = V(z) + H * (P(epsilon) * V(z) * (P(alpha) + V(x) * V(y)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * 0.5 * (V(y) * (V(z) - P(betta) + pow(V(x), 2)) + P(gamma) * V(x));
        numb ymp = V(y) + H * 0.5 * (V(x) * (P(delta) * V(z) + P(betta) - pow(V(x), 2)) + P(gamma) * V(y));
        numb zmp = V(z) + H * 0.5 * (P(epsilon) * V(z) * (P(alpha) + V(x) * V(y)));

        Vnext(x) = V(x) + H * (ymp * (zmp - P(betta) + pow(xmp, 2)) + P(gamma) * xmp);
        Vnext(y) = V(y) + H * (xmp * (P(delta) * zmp + P(betta) - pow(xmp, 2)) + P(gamma) * ymp);
        Vnext(z) = V(z) + H * (P(epsilon) * zmp * (P(alpha) + xmp * ymp));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = V(y) * (V(z) - P(betta) + pow(V(x), 2)) + P(gamma) * V(x);
        numb ky1 = V(x) * (P(delta) * V(z) + P(betta) - pow(V(x), 2)) + P(gamma) * V(y);
        numb kz1 = P(epsilon) * V(z) * (P(alpha) + V(x) * V(y));

        numb xmp = V(x) + 0.5 * H * kx1;
        numb ymp = V(y) + 0.5 * H * ky1;
        numb zmp = V(z) + 0.5 * H * kz1;

        numb kx2 = ymp * (zmp - P(betta) + pow(xmp, 2)) + P(gamma) * xmp;
        numb ky2 = xmp * (P(delta) * zmp + P(betta) - pow(xmp, 2)) + P(gamma) * ymp;
        numb kz2 = P(epsilon) * zmp * (P(alpha) + xmp * ymp);

        xmp = V(x) + 0.5 * H * kx2;
        ymp = V(y) + 0.5 * H * ky2;
        zmp = V(z) + 0.5 * H * kz2;

        numb kx3 = ymp * (zmp - P(betta) + pow(xmp, 2)) + P(gamma) * xmp;
        numb ky3 = xmp * (P(delta) * zmp + P(betta) - pow(xmp, 2)) + P(gamma) * ymp;
        numb kz3 = P(epsilon) * zmp * (P(alpha) + xmp * ymp);

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = ymp * (zmp - P(betta) + pow(xmp, 2)) + P(gamma) * xmp;
        numb ky4 = xmp * (P(delta) * zmp + P(betta) - pow(xmp, 2)) + P(gamma) * ymp;
        numb kz4 = P(epsilon) * zmp * (P(alpha) + xmp * ymp);

        Vnext(x) = V(x) + H * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4) / 6.0;
        Vnext(y) = V(y) + H * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4) / 6.0;
        Vnext(z) = V(z) + H * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4) / 6.0;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (V(y) * (V(z) - P(betta) + pow(V(x), 2)) + P(gamma) * V(x));
        numb ymp = V(y) + h1 * (xmp * (P(delta) * V(z) + P(betta) - pow(xmp, 2)) + P(gamma) * V(y));
        numb zmp = V(z) + h1 * (P(epsilon) * V(z) * (P(alpha) + xmp * ymp));

        numb denom_z = 1.0 - h2 * P(epsilon) * (P(alpha) + xmp * ymp);
        if (fabs(denom_z) < 1e-6) denom_z = copysign(1e-6, denom_z);
        Vnext(z) = zmp / denom_z;

        numb denom_y = 1.0 - h2 * P(gamma);
        if (fabs(denom_y) < 1e-6) denom_y = copysign(1e-6, denom_y);
        Vnext(y) = (ymp + h2 * xmp * (P(delta) * Vnext(z) + P(betta) - pow(xmp, 2))) / denom_y;

        Vnext(x) = xmp + h2 * (Vnext(y) * (Vnext(z) - P(betta) + pow(xmp, 2)) + P(gamma) * xmp);
        Vnext(x) = xmp + h2 * (Vnext(y) * (Vnext(z) - P(betta) + pow(Vnext(x), 2)) + P(gamma) * Vnext(x));
    }
}

#undef name