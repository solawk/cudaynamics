#include "main.h"
#include "sprottJm.h"

#define name sprottJm

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, method, COUNT };
    enum methods { ExplicitEuler, ExplicitRungeKutta4 };
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

__host__ __device__ numb sprottJm_F(numb y)
{
    numb b, k;
    numb d = 1.0, m = 1.0, P = 1.23, R = 2.0;
    numb ay = abs(y);
    while (true)
    {
        if (ay < d)
        {
            d /= R; m /= R;
        }
        else if (ay > 2 * d)
        {
            d *= R; m *= R;
        }
        else
            break;
    }
    numb epsilon = 0.01;
    if (d > epsilon)
    {
        if (ay < P * d)
        {
            b = -m * (R - P * R + 1) / (R * (P - 1));
            k = -m / (R * d * (1 - P));
        }
        else
        {
            b = -m * (R - P * R + 1) / (P - R);
            k = -m * (-(R * R) + R + 1) / (R * d * (P - R));
        }

        return k * ay + b;
    }
    else
        return 0.0;
}

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters)
{  
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + H * (P(a) * V(z));
        Vnext(y) = V(y) + H * (P(b) * V(y) + V(z));
        Vnext(z) = V(z) + H * (-V(x) + V(y) + P(c) * sprottJm_F(V(y)));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(a) * V(z);
        numb ky1 = P(b) * V(y) + V(z);
        numb kz1 = -V(x) + V(y) + P(c) * sprottJm_F(V(y));

        numb xmp = V(x) + 0.5 * H * kx1;
        numb ymp = V(y) + 0.5 * H * ky1;
        numb zmp = V(z) + 0.5 * H * kz1;

        numb kx2 = P(a) * zmp;
        numb ky2 = P(b) * ymp + zmp;
        numb kz2 = -xmp + ymp + P(c) * sprottJm_F(ymp);

        xmp = V(x) + 0.5 * H * kx2;
        ymp = V(y) + 0.5 * H * ky2;
        zmp = V(z) + 0.5 * H * kz2;

        numb kx3 = P(a) * zmp;
        numb ky3 = P(b) * ymp + zmp;
        numb kz3 = -xmp + ymp + P(c) * sprottJm_F(ymp);

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = P(a) * zmp;
        numb ky4 = P(b) * ymp + zmp;
        numb kz4 = -xmp + ymp + P(c) * sprottJm_F(ymp);

        Vnext(x) = V(x) + H * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;
        Vnext(y) = V(y) + H * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6;
        Vnext(z) = V(z) + H * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6;
    }
}

#undef name