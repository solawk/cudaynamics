#include "rossler.h"

#define name rossler

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { a, b, c, symmetry, method, COUNT };
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
    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_(name));

    int steps = !CUDA_kernel.usingTime ? CUDA_kernel.steps : CUDA_kernel.time / H;
    bool noDecimation = CUDA_kernel.targetSteps == steps;
   
    if (noDecimation)
    {
        for (int s = 0; s < steps && !data->isHires; s++)
        {
            finiteDifferenceScheme_(name)(FDS_ARGUMENTS);
            stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
            RECORD_STEP;
        }
    }
    else
    {
        numb sfloat = (numb)0.0;
        numb sfloatStep = (numb)CUDA_kernel.targetSteps / steps;
        int scounter = 0;

        for (int s = 0; s < steps && !data->isHires; s++)
        {
            finiteDifferenceScheme_(name)(FDS_ARGUMENTS);
            TRANSFER_VARIABLES;
            sfloat += sfloatStep;
            if (sfloat >= (numb)1.0 || s == steps - 1)
            {
                sfloat -= floor(sfloat);
                stepStart = variationStart + scounter * CUDA_kernel.VAR_COUNT;
                RECORD_STEP_WO_NEXT;
                scounter++;
            }
        }
    }

    // Analysis
    AnalysisLobby(data, &finiteDifferenceScheme_(name), variation);
}

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + H * (-(V(y) + V(z)));
        Vnext(y) = V(y) + H * (V(x) + P(a) * V(y));
        Vnext(z) = V(z) + H * (P(b) + V(z) * (V(x) - P(c)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * (numb)0.5 * (-(V(y) + V(z)));
        numb ymp = V(y) + H * (numb)0.5 * (V(x) + P(a) * V(y));
        numb zmp = V(z) + H * (numb)0.5 * (P(b) + V(z) * (V(x) - P(c)));

        Vnext(x) = V(x) + H * (-(ymp + zmp));
        Vnext(y) = V(y) + H * (xmp + P(a) * ymp);
        Vnext(z) = V(z) + H * (P(b) + zmp * (xmp - P(c)));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = -(V(y) + V(z));
        numb ky1 = V(x) + P(a) * V(y);
        numb kz1 = P(b) + V(z) * (V(x) - P(c));

        numb xmp = V(x) + (numb)0.5 * H * kx1;
        numb ymp = V(y) + (numb)0.5 * H * ky1;
        numb zmp = V(z) + (numb)0.5 * H * kz1;

        numb kx2 = -(ymp + zmp);
        numb ky2 = xmp + P(a) * ymp;
        numb kz2 = P(b) + zmp * (xmp - P(c));

        xmp = V(x) + (numb)0.5 * H * kx2;
        ymp = V(y) + (numb)0.5 * H * ky2;
        zmp = V(z) + (numb)0.5 * H * kz2;

        numb kx3 = -(ymp + zmp);
        numb ky3 = xmp + P(a) * ymp;
        numb kz3 = P(b) + zmp * (xmp - P(c));

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = -(ymp + zmp);
        numb ky4 = xmp + P(a) * ymp;
        numb kz4 = P(b) + zmp * (xmp - P(c));

        Vnext(x) = V(x) + H * (kx1 + (numb)2 * kx2 + (numb)2 * kx3 + kx4) / (numb)6;
        Vnext(y) = V(y) + H * (ky1 + (numb)2 * ky2 + (numb)2 * ky3 + ky4) / (numb)6;
        Vnext(z) = V(z) + H * (kz1 + (numb)2 * kz2 + (numb)2 * kz3 + kz4) / (numb)6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = (numb)0.5 * H - P(symmetry);
        numb h2 = (numb)0.5 * H + P(symmetry);

        numb zmp = V(z) + h1 * (P(b) + V(z) * (V(x) - P(c)));
        numb ymp = V(y) + h1 * (V(x) + P(a) * V(y));
        numb xmp = V(x) + h1 * (-(ymp + zmp));
        
        Vnext(x) = xmp + h2 * (-(ymp + zmp));
        Vnext(y) = (ymp + h2 * Vnext(x)) / (1 - P(a) * h2);
        Vnext(z) = (zmp + h2 * P(b)) / (1 - (Vnext(x) - P(c)) * h2);
    }
}

#undef name