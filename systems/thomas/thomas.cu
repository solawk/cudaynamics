#include "main.h"
#include "thomas.h"

#define name thomas

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { b, symmetry, method, COUNT };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
}

__global__ void kernelProgram_(name)(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
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

__device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + H * (sin(V(y)) - P(b) * V(x));
        Vnext(y) = V(y) + H * (sin(V(z)) - P(b) * V(y));
        Vnext(z) = V(z) + H * (sin(V(x)) - P(b) * V(z));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * 0.5 * (sin(V(y)) - P(b) * V(x));
        numb ymp = V(y) + H * 0.5 * (sin(V(z)) - P(b) * V(y));
        numb zmp = V(z) + H * 0.5 * (sin(V(x)) - P(b) * V(z));

        Vnext(x) = V(x) + H * (sin(ymp) - P(b) * xmp);
        Vnext(y) = V(y) + H * (sin(zmp) - P(b) * ymp);
        Vnext(z) = V(z) + H * (sin(xmp) - P(b) * zmp);
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = sin(V(y)) - P(b) * V(x);
        numb ky1 = sin(V(z)) - P(b) * V(y);
        numb kz1 = sin(V(x)) - P(b) * V(z);

        numb xmp = V(x) + 0.5 * H * kx1;
        numb ymp = V(y) + 0.5 * H * ky1;
        numb zmp = V(z) + 0.5 * H * kz1;

        numb kx2 = sin(ymp) - P(b) * xmp;
        numb ky2 = sin(zmp) - P(b) * ymp;
        numb kz2 = sin(xmp) - P(b) * zmp;

        xmp = V(x) + 0.5 * H * kx2;
        ymp = V(y) + 0.5 * H * ky2;
        zmp = V(z) + 0.5 * H * kz2;

        numb kx3 = sin(ymp) - P(b) * xmp;
        numb ky3 = sin(zmp) - P(b) * ymp;
        numb kz3 = sin(xmp) - P(b) * zmp;

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = sin(ymp) - P(b) * xmp;
        numb ky4 = sin(zmp) - P(b) * ymp;
        numb kz4 = sin(xmp) - P(b) * zmp;

        Vnext(x) = V(x) + H * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4) / 6.0;
        Vnext(y) = V(y) + H * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4) / 6.0;
        Vnext(z) = V(z) + H * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4) / 6.0;
    }
    ifMETHOD(P(method), VariableSymmetryCD)
    {  
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (sin(V(y)) - P(b) * V(x));
        numb ymp = V(y) + h1 * (sin(V(z)) - P(b) * V(y));
        numb zmp = V(z) + h1 * (sin(xmp) - P(b) * V(z));

        numb denom_z = (1 + h2 * P(b));
        if (fabs(denom_z) < 1e-6) denom_z = copysign(1e-6, denom_z);
        Vnext(z) = (zmp + h2 * sin(xmp)) / denom_z;

        numb denom_y = (1 + h2 * P(b));
        if (fabs(denom_y) < 1e-6) denom_y = copysign(1e-6, denom_y);
        Vnext(y) = (ymp + h2 * sin(Vnext(z))) / denom_y;

        numb denom_x = (1 + h2 * P(b));
        if (fabs(denom_x) < 1e-6) denom_x = copysign(1e-6, denom_x);
        Vnext(x) = (xmp + h2 * sin(Vnext(y))) / denom_x;
    }
}

#undef name