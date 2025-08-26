#include "main.h"
#include "jj_rlcs.h"

namespace attributes
{
    enum variables { x1, sin_x1, x2, x3 };
    enum parameters { betaL, betaC, i, gThreshold, Rn, Rsg, stepsize, symmetry, method };
    enum methods { ExplicitEuler, SemiExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE };
}

__global__ void kernelProgram_jj_rlcs(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step
    //int indicesStart = variation * (CUDA_kernel.VAR_COUNT + CUDA_kernel.PARAM_COUNT);   // Start index for the step indices of the attributes in the current variation

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_jj_rlcs);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_jj_rlcs(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]));
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_jj_rlcs, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_jj_rlcs(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x1) = fmodf(V(x1) + P(stepsize) * V(x2), 2.0f * 3.141592f);
        Vnext(sin_x1) = sinf(Vnext(x1));
        Vnext(x2) = V(x2) + P(stepsize) * (1.0f / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sinf(V(x1)) - V(x3));
        Vnext(x3) = V(x3) + P(stepsize) * (1.0f / P(betaL)) * (V(x2) - V(x3));
    }

    ifMETHOD(P(method), SemiExplicitEuler)
    {
        Vnext(x1) = fmodf(V(x1) + P(stepsize) * V(x2), 2.0f * 3.141592f);
        Vnext(sin_x1) = sinf(Vnext(x1));
        Vnext(x3) = V(x3) + P(stepsize) * (1.0f / P(betaL)) * (V(x2) - V(x3));
        Vnext(x2) = V(x2) + P(stepsize) * (1.0f / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sinf(Vnext(x1)) - Vnext(x3));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb x1mp = fmodf(V(x1) + P(stepsize) * 0.5 * V(x2), 2.0f * 3.141592f);
        numb x2mp = V(x2) + P(stepsize) * 0.5 * (1.0f / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sinf(V(x1)) - V(x3));
        numb x3mp = V(x3) + P(stepsize) * 0.5 * (1.0f / P(betaL)) * (V(x2) - V(x3));

        Vnext(x1) = fmodf(V(x1) + P(stepsize) * x2mp, 2.0f * 3.141592f);
        Vnext(sin_x1) = sinf(Vnext(x1));
        Vnext(x2) = V(x2) + P(stepsize) * (1.0f / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sinf(x1mp) - x3mp);
        Vnext(x3) = V(x3) + P(stepsize) * (1.0f / P(betaL)) * (x2mp - x3mp);
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx11 = V(x2);
        numb kx21 = (1.0f / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sinf(V(x1)) - V(x3));
        numb kx31 = (1.0f / P(betaL)) * (V(x2) - V(x3));

        numb x1mp = fmodf(V(x1) + P(stepsize) * 0.5 * kx11, 2.0f * 3.141592f);
        numb x2mp = V(x2) + P(stepsize) * 0.5 * kx21;
        numb x3mp = V(x3) + P(stepsize) * 0.5 * kx31;

        numb kx12 = x2mp;
        numb kx22 = (1.0f / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sinf(x1mp) - x3mp);
        numb kx32 = (1.0f / P(betaL)) * (x2mp - x3mp);

        x1mp = fmodf(V(x1) + P(stepsize) * 0.5 * kx12, 2.0f * 3.141592f);
        x2mp = V(x2) + P(stepsize) * 0.5 * kx22;
        x3mp = V(x3) + P(stepsize) * 0.5 * kx32;

        numb kx13 = x2mp;
        numb kx23 = (1.0f / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sinf(x1mp) - x3mp);
        numb kx33 = (1.0f / P(betaL)) * (x2mp - x3mp);

        x1mp = fmodf(V(x1) + P(stepsize) * kx13, 2.0f * 3.141592f);
        x2mp = V(x2) + P(stepsize) * kx23;
        x3mp = V(x3) + P(stepsize) * kx33;

        numb kx14 = x2mp;
        numb kx24 = (1.0f / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sinf(x1mp) - x3mp);
        numb kx34 = (1.0f / P(betaL)) * (x2mp - x3mp);

        Vnext(x1) = fmodf(V(x1) + P(stepsize) * (kx11 + 2 * kx12 + 2 * kx13 + kx14) / 6, 2.0f * 3.141592f);
        Vnext(sin_x1) = sinf(Vnext(x1));
        Vnext(x2) = V(x2) + P(stepsize) * (kx21 + 2 * kx22 + 2 * kx23 + kx24) / 6;
        Vnext(x3) = V(x3) + P(stepsize) * (kx31 + 2 * kx32 + 2 * kx33 + kx34) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * P(stepsize) - P(symmetry);
        numb h2 = 0.5 * P(stepsize) + P(symmetry);

        numb x1mp = fmodf(V(x1) + h1 * V(x2), 2.0f * 3.141592f);
        numb x3mp = V(x3) + h1 * (1.0f / P(betaL)) * (V(x2) - V(x3));
        numb x2mp = V(x2) + h1 * (1.0f / P(betaC)) * (P(i) - ((V(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * V(x2) - sinf(x1mp) - x3mp);

        Vnext(x2) = x2mp + h2 * (1.0f / P(betaC)) * (P(i) - ((x2mp > P(gThreshold)) ? P(Rn) : P(Rsg)) * x2mp - sinf(x1mp) - x3mp);
        Vnext(x2) = x2mp + h2 * (1.0f / P(betaC)) * (P(i) - ((Vnext(x2) > P(gThreshold)) ? P(Rn) : P(Rsg)) * Vnext(x2) - sinf(x1mp) - x3mp);
        Vnext(x3) = (x3mp + h2 * (1.0f / P(betaL)) * Vnext(x2)) / (1.0f + h2 * (1.0f / P(betaL)));
        Vnext(x1) = fmodf(x1mp + h2 * Vnext(x2), 2.0f * 3.141592f);
        Vnext(sin_x1) = sinf(Vnext(x1));
    }
}
