#include "lorenz84.h"

#define name lorenz84

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, f, g, symmetry, method, COUNT };
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
        Vnext(x) = V(x) + H * (-P(a) * V(x) - pow(V(y), (numb)2.0) - pow(V(z), (numb)2.0) + P(a) * P(f));
        Vnext(y) = V(y) + H * (-V(y) + V(x) * V(y) - P(b) * V(x) * V(z) + P(g));
        Vnext(z) = V(z) + H * (-V(z) + P(b) * V(x) * V(y) + V(x) * V(z));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * (numb)0.5 * (-P(a) * V(x) - pow(V(y), (numb)2.0) - pow(V(z), (numb)2.0) + P(a) * P(f));
        numb ymp = V(y) + H * (numb)0.5 * (-V(y) + V(x) * V(y) - P(b) * V(x) * V(z) + P(g));
        numb zmp = V(z) + H * (numb)0.5 * (-V(z) + P(b) * V(x) * V(y) + V(x) * V(z));

        Vnext(x) = V(x) + H * (-P(a) * xmp - pow(ymp, (numb)2.0) - pow(zmp, (numb)2.0) + P(a) * P(f));
        Vnext(y) = V(y) + H * (-ymp + xmp * ymp - P(b) * xmp * zmp + P(g));
        Vnext(z) = V(z) + H * (-zmp + P(b) * xmp * ymp + xmp * zmp);
    }
    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = -P(a) * V(x) - pow(V(y), (numb)2.0) - pow(V(z), (numb)2.0) + P(a) * P(f);
        numb ky1 = -V(y) + V(x) * V(y) - P(b) * V(x) * V(z) + P(g);
        numb kz1 = -V(z) + P(b) * V(x) * V(y) + V(x) * V(z);

        numb xmp = V(x) + (numb)0.5 * H * kx1;
        numb ymp = V(y) + (numb)0.5 * H * ky1;
        numb zmp = V(z) + (numb)0.5 * H * kz1;

        numb kx2 = -P(a) * xmp - pow(ymp, (numb)2.0) - pow(zmp, (numb)2.0) + P(a) * P(f);
        numb ky2 = -ymp + xmp * ymp - P(b) * xmp * zmp + P(g);
        numb kz2 = -zmp + P(b) * xmp * ymp + xmp * zmp;

        xmp = V(x) + (numb)0.5 * H * kx2;
        ymp = V(y) + (numb)0.5 * H * ky2;
        zmp = V(z) + (numb)0.5 * H * kz2;

        numb kx3 = -P(a) * xmp - pow(ymp, (numb)2.0) - pow(zmp, (numb)2.0) + P(a) * P(f);
        numb ky3 = -ymp + xmp * ymp - P(b) * xmp * zmp + P(g);
        numb kz3 = -zmp + P(b) * xmp * ymp + xmp * zmp;

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = -P(a) * xmp - pow(ymp, (numb)2.0) - pow(zmp, (numb)2.0) + P(a) * P(f);
        numb ky4 = -ymp + xmp * ymp - P(b) * xmp * zmp + P(g);
        numb kz4 = -zmp + P(b) * xmp * ymp + xmp * zmp;

        Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
        Vnext(y) = V(y) + H * (ky1 + (numb)2.0 * ky2 + (numb)2.0 * ky3 + ky4) / (numb)6.0;
        Vnext(z) = V(z) + H * (kz1 + (numb)2.0 * kz2 + (numb)2.0 * kz3 + kz4) / (numb)6.0;
    }
    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = (numb)0.5 * H - P(symmetry);
        numb h2 = (numb)0.5 * H + P(symmetry);

        numb xmp = V(x) + (numb)0.5 * h1 * (-P(a) * V(x) - pow(V(y), (numb)2.0) - pow(V(z), (numb)2.0) + P(a) * P(f));
        numb ymp = V(y) + (numb)0.5 * h1 * (-V(y) + xmp * V(y) - P(b) * xmp * V(z) + P(g));
        numb zmp = V(z) + (numb)0.5 * h1 * (-V(z) + P(b) * xmp * ymp + xmp * V(z));

        numb denom_z = ((numb)1.0 + h2 - h2 * xmp);
        if (fabs(denom_z) < (numb)1e-6) denom_z = copysign((numb)1e-6, denom_z);
        Vnext(z) = (zmp + h2 * P(b) * xmp * ymp) / denom_z;

        numb denom_y = ((numb)1.0 + h2 - h2 * xmp);
        if (fabs(denom_y) < (numb)1e-6) denom_y = copysign((numb)1e-6, denom_y);
        Vnext(y) = (ymp - h2 * P(b) * xmp * Vnext(z) + h2 * P(g)) / denom_y;

        numb denom_x = ((numb)1.0 + h2 * P(a));
        if (fabs(denom_x) < (numb)1e-6) denom_x = copysign((numb)1e-6, denom_x);
        Vnext(x) = (xmp - h2 * pow(Vnext(y), (numb)2.0) - h2 * pow(Vnext(z), (numb)2.0) + h2 * P(a) * P(f)) / denom_x;
    }
}

#undef name