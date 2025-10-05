#include "main.h"
#include "lorenz.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { sigma, rho, beta, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE, MAX };
}

__global__ void kernelProgram_lorenz(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart;                                              // Start index for the current modelling step 
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++) variables[i] = CUDA_marshal.trajectory[variationStart + i];
    for (int i = 0; i < CUDA_kernel.PARAM_COUNT; i++) parameters[i] = CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT + i];

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_lorenz);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_lorenz(&(variables[0]), &(variablesNext[0]), &(parameters[0]));

        for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
        {
            CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT + i] = variables[i] = variablesNext[i];
        }
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_lorenz, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_lorenz, MO(MAX));
    }
}

__device__ __forceinline__ void finiteDifferenceScheme_lorenz(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (P(sigma) * (V(y) - V(x)));
        Vnext(y) = V(y) + P(stepsize) * (V(x) * (P(rho) - V(z)) - V(y));
        Vnext(z) = V(z) + P(stepsize) * (V(x) * V(y) - P(beta) * V(z));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + P(stepsize) * 0.5f * (P(sigma) * (V(y) - V(x)));
        numb ymp = V(y) + P(stepsize) * 0.5f * (V(x) * (P(rho) - V(z)) - V(y));
        numb zmp = V(z) + P(stepsize) * 0.5f * (V(x) * V(y) - P(beta) * V(z));

        Vnext(x) = V(x) + P(stepsize) * (P(sigma) * (ymp - xmp));
        Vnext(y) = V(y) + P(stepsize) * (xmp * (P(rho) - zmp) - ymp);
        Vnext(z) = V(z) + P(stepsize) * (xmp * ymp - P(beta) * zmp);
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(sigma) * (V(y) - V(x));
        numb ky1 = V(x) * (P(rho) - V(z)) - V(y);
        numb kz1 = V(x) * V(y) - P(beta) * V(z);

        numb xmp = V(x) + 0.5f * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5f * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5f * P(stepsize) * kz1;

        numb kx2 = P(sigma) * (ymp - xmp);
        numb ky2 = xmp * (P(rho) - zmp) - ymp;
        numb kz2 = xmp * ymp - P(beta) * zmp;

        xmp = V(x) + 0.5f * P(stepsize) * kx2;
        ymp = V(y) + 0.5f * P(stepsize) * ky2;
        zmp = V(z) + 0.5f * P(stepsize) * kz2;

        numb kx3 = P(sigma) * (ymp - xmp);
        numb ky3 = xmp * (P(rho) - zmp) - ymp;
        numb kz3 = xmp * ymp - P(beta) * zmp;

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = P(sigma) * (ymp - xmp);
        numb ky4 = xmp * (P(rho) - zmp) - ymp;
        numb kz4 = xmp * ymp - P(beta) * zmp;

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2.0f * kx2 + 2.0f * kx3 + kx4) / 6.0f;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2.0f * ky2 + 2.0f * ky3 + ky4) / 6.0f;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2.0f * kz2 + 2.0f * kz3 + kz4) / 6.0f;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5f * P(stepsize) - P(symmetry);
        numb h2 = 0.5f * P(stepsize) + P(symmetry);

        numb xmp = V(x) + h1 * (P(sigma) * (V(y) - V(x)));
        numb ymp = V(y) + h1 * (V(x) * (P(rho) - V(z)) - V(y));
        numb zmp = V(z) + h1 * (V(x) * V(y) - P(beta) * V(z));

        Vnext(z) = (zmp + xmp * ymp * h2) / (1.0f + P(beta) * h2);
        Vnext(y) = (ymp + xmp * (P(rho) - Vnext(z)) * h2) / (1.0f + h2);
        Vnext(x) = (xmp + P(sigma) * Vnext(y) * h2) / (1.0f + P(sigma) * h2);
    }
}