#include "main.h"
#include "jj_mrlcs.h"

namespace attributes
{
    enum variables { theta, sin_theta, v, iL };
    enum parameters { betaL, betaC, betaM, i, epsilon, delta, stepsize, symmetry, method };
    enum methods { ExplicitEuler, SemiExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE };
}

__global__ void kernelProgram_jj_mrlcs(Computation* data)
{
    int b = blockIdx.x;                                     // Current block of THREADS_PER_BLOCK threads
    int t = threadIdx.x;                                    // Current thread in the block, from 0 to THREADS_PER_BLOCK-1
    int variation = (b * data->threads_per_block) + t;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step
    //int indicesStart = variation * (CUDA_kernel.VAR_COUNT + CUDA_kernel.PARAM_COUNT);   // Start index for the step indices of the attributes in the current variation

    // Custom area (usually) starts here

    TRANSIENT_SKIP(finiteDifferenceScheme_jj_mrlcs);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_jj_mrlcs(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]),
            CUDA_kernel.stepSize);
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_jj_mrlcs, MO(LLE));
    }
}

__device__ void finiteDifferenceScheme_jj_mrlcs(numb* currentV, numb* nextV, numb* parameters, numb h)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(theta) = fmodf(V(theta) + P(stepsize) * V(v), 2.0f * 3.141592f);
        Vnext(sin_theta) = sinf(Vnext(theta));
        Vnext(iL) = V(iL) + P(stepsize) * ((1.0f / P(betaL)) * (V(v) - V(iL)));
        Vnext(v) = V(v) + P(stepsize) * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));
    }

    ifMETHOD(P(method), SemiExplicitEuler)
    {
        Vnext(theta) = fmodf(V(theta) + P(stepsize) * V(v), 2.0f * 3.141592f);
        Vnext(sin_theta) = sinf(Vnext(theta));
        Vnext(iL) = V(iL) + P(stepsize) * ((1.0f / P(betaL)) * (V(v) - V(iL)));
        Vnext(v) = V(v) + P(stepsize) * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(Vnext(theta))) * V(v) - sinf(Vnext(theta)) - P(delta) * Vnext(iL)));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb thetamp = fmodf(V(theta) + P(stepsize) * 0.5 * V(v), 2.0f * 3.141592f);
        numb iLmp = V(iL) + P(stepsize) * 0.5 * ((1.0f / P(betaL)) * (V(v) - V(iL)));
        numb vmp = V(v) + P(stepsize) * 0.5 * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));

        Vnext(theta) = fmodf(V(theta) + P(stepsize) * vmp, 2.0f * 3.141592f);
        Vnext(sin_theta) = sinf(Vnext(theta));
        Vnext(iL) = V(iL) + P(stepsize) * ((1.0f / P(betaL)) * (vmp - iLmp));
        Vnext(v) = V(v) + P(stepsize) * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb ktheta1 = V(v);
        numb kiL1 = ((1.0f / P(betaL)) * (V(v) - V(iL)));
        numb kv1 = ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));

        numb thetamp = fmodf(V(theta) + P(stepsize) * 0.5 * ktheta1, 2.0f * 3.141592f);
        numb iLmp = V(iL) + P(stepsize) * 0.5 * kiL1;
        numb vmp = V(v) + P(stepsize) * 0.5 * kv1;

        numb ktheta2 = vmp;
        numb kiL2 = ((1.0f / P(betaL)) * (vmp - iLmp));
        numb kv2 = ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

        thetamp = fmodf(V(theta) + P(stepsize) * 0.5 * ktheta2, 2.0f * 3.141592f);
        iLmp = V(iL) + P(stepsize) * 0.5 * kiL2;
        vmp = V(v) + P(stepsize) * 0.5 * kv2;

        numb ktheta3 = vmp;
        numb kiL3 = ((1.0f / P(betaL)) * (vmp - iLmp));
        numb kv3 = ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

        thetamp = fmodf(V(theta) + P(stepsize) * ktheta3, 2.0f * 3.141592f);
        iLmp = V(iL) + P(stepsize) * kiL3;
        vmp = V(v) + P(stepsize) * kv3;

        numb ktheta4 = vmp;
        numb kiL4 = ((1.0f / P(betaL)) * (vmp - iLmp));
        numb kv4 = ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

        Vnext(theta) = fmodf(V(theta) + P(stepsize) * (ktheta1 + 2 * ktheta2 + 2 * ktheta3 + ktheta4) / 6, 2.0f * 3.141592f);
        Vnext(sin_theta) = sinf(Vnext(theta));
        Vnext(iL) = V(iL) + P(stepsize) * (kiL1 + 2 * kiL2 + 2 * kiL3 + kiL4) / 6;
        Vnext(v) = V(v) + P(stepsize) * (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * P(stepsize) - P(symmetry);
        numb h2 = 0.5 * P(stepsize) + P(symmetry);

        numb thetamp = fmodf(V(theta) + h1 * V(v), 2.0f * 3.141592f);
        numb iLmp = V(iL) + h1 * ((1.0f / P(betaL)) * (V(v) - V(iL)));
        numb vmp = V(v) + h1 * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * V(v) - sinf(thetamp) - P(delta) * iLmp));

        Vnext(v) = vmp + h2 * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));
        Vnext(v) = vmp + h2 * ((1.0f / P(betaC)) * (P(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * Vnext(v) - sinf(thetamp) - P(delta) * iLmp));
        Vnext(iL) = (iLmp + h2 * (1.0f / P(betaL)) * Vnext(v)) / (1 + h2 * (1.0f / P(betaL)));
        Vnext(theta) = fmodf(thetamp + h2 * Vnext(v), 2.0f * 3.141592f);
        Vnext(sin_theta) = sinf(Vnext(theta));
    }
}