#include "main.h"
#include "lorenzVar.h"

#define name lorenzVar

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { sigma, rho, betta, kappa, gamma, symmetry, method, COUNT };
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
        Vnext(x) = V(x) + H * (P(sigma) * (-V(x) + V(y)) + P(kappa) * sin(V(y) / P(gamma)) * sin(V(z) / P(gamma)));
        Vnext(y) = V(y) + H * (-V(x) * V(z) + P(rho) * V(x) - V(y) + P(kappa) * sin(V(x) / P(gamma)) * sin(V(z) / P(gamma)));
        Vnext(z) = V(z) + H * (V(x) * V(y) - P(betta) * V(z) + P(kappa) * cos(V(y) / P(gamma)) * cos(V(x) / P(gamma)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * 0.5 * (P(sigma) * (-V(x) + V(y)) + P(kappa) * sin(V(y) / P(gamma)) * sin(V(z) / P(gamma)));
        numb ymp = V(y) + H * 0.5 * (-V(x) * V(z) + P(rho) * V(x) - V(y) + P(kappa) * sin(V(x) / P(gamma)) * sin(V(z) / P(gamma)));
        numb zmp = V(z) + H * 0.5 * (V(x) * V(y) - P(betta) * V(z) + P(kappa) * cos(V(y) / P(gamma)) * cos(V(x) / P(gamma)));

        Vnext(x) = V(x) + H * (P(sigma) * (-xmp + ymp) + P(kappa) * sin(ymp / P(gamma)) * sin(zmp / P(gamma)));
        Vnext(y) = V(y) + H * (-xmp * zmp + P(rho) * xmp - ymp + P(kappa) * sin(xmp / P(gamma)) * sin(zmp / P(gamma)));
        Vnext(z) = V(z) + H * (xmp * ymp - P(betta) * zmp + P(kappa) * cos(ymp / P(gamma)) * cos(xmp / P(gamma)));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(sigma) * (-V(x) + V(y)) + P(kappa) * sin(V(y) / P(gamma)) * sin(V(z) / P(gamma));
        numb ky1 = -V(x) * V(z) + P(rho) * V(x) - V(y) + P(kappa) * sin(V(x) / P(gamma)) * sin(V(z) / P(gamma));
        numb kz1 = V(x) * V(y) - P(betta) * V(z) + P(kappa) * cos(V(y) / P(gamma)) * cos(V(x) / P(gamma));

        numb xmp = V(x) + 0.5 * H * kx1;
        numb ymp = V(y) + 0.5 * H * ky1;
        numb zmp = V(z) + 0.5 * H * kz1;

        numb kx2 = P(sigma) * (-xmp + ymp) + P(kappa) * sin(ymp / P(gamma)) * sin(zmp / P(gamma));
        numb ky2 = -xmp * zmp + P(rho) * xmp - ymp + P(kappa) * sin(xmp / P(gamma)) * sin(zmp / P(gamma));
        numb kz2 = xmp * ymp - P(betta) * zmp + P(kappa) * cos(ymp / P(gamma)) * cos(zmp / P(gamma));

        xmp = V(x) + 0.5 * H * kx2;
        ymp = V(y) + 0.5 * H * ky2;
        zmp = V(z) + 0.5 * H * kz2;

        numb kx3 = P(sigma) * (-xmp + ymp) + P(kappa) * sin(ymp / P(gamma)) * sin(zmp / P(gamma));
        numb ky3 = -xmp * zmp + P(rho) * xmp - ymp + P(kappa) * sin(xmp / P(gamma)) * sin(zmp / P(gamma));
        numb kz3 = xmp * ymp - P(betta) * zmp + P(kappa) * cos(ymp / P(gamma)) * cos(zmp / P(gamma));

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = P(sigma) * (-xmp + ymp) + P(kappa) * sin(ymp / P(gamma)) * sin(zmp / P(gamma));
        numb ky4 = -xmp * zmp + P(rho) * xmp - ymp + P(kappa) * sin(xmp / P(gamma)) * sin(zmp / P(gamma));
        numb kz4 = xmp * ymp - P(betta) * zmp + P(kappa) * cos(ymp / P(gamma)) * cos(zmp / P(gamma));

        Vnext(x) = V(x) + H * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4) / 6.0;
        Vnext(y) = V(y) + H * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4) / 6.0;
        Vnext(z) = V(z) + H * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4) / 6.0;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb x1 = V(x) + h1 * (P(sigma) * (-V(x) + V(y)) + P(kappa) * sin(V(y) / P(gamma)) * sin(V(z) / P(gamma)));
        numb y1 = V(y) + h1 * (-V(x) * V(z) + P(rho) * V(x) - V(y) + P(kappa) * sin(V(x) / P(gamma)) * sin(V(z) / P(gamma)));
        numb z1 = V(z) + h1 * (V(x) * V(y) - P(betta) * V(z) + P(kappa) * cos(V(y) / P(gamma)) * cos(V(x) / P(gamma)));

        numb denom_z = (1 + h2 * P(betta));
        if (fabs(denom_z) < 1e-6) denom_z = copysign(1e-6, denom_z);
        Vnext(z) = (z1 + h2 * (x1 * y1 + P(kappa) * cos(y1 / P(gamma)) * cos(x1 / P(gamma)))) / denom_z;

        numb denom_y = (1 + h2 * P(betta));
        if (fabs(denom_y) < 1e-6) denom_y = copysign(1e-6, denom_y);
        Vnext(y) = (y1 + h2 * (-x1 * Vnext(z) + P(rho) * x1 + P(kappa) * sin(x1 / P(gamma)) * sin(Vnext(z) / P(gamma)))) / denom_y;

        numb denom_x = (1 + h2 * P(sigma));
        if (fabs(denom_x) < 1e-6) denom_x = copysign(1e-6, denom_x);
        Vnext(x) = (x1 + h2 * (P(sigma) * Vnext(y) + P(kappa) * sin(Vnext(y) / P(gamma)) * sin(Vnext(z) / P(gamma)))) / denom_x;
    }
}

#undef name