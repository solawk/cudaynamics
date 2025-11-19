#include "main.h"
#include "jj_mrlcs.h"

#define name jj_mrlcs

namespace attributes
{
    enum variables { theta, sin_theta, v, iL, i, t };
    enum parameters { betaL, betaC, betaM, epsilon, delta, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, method, COUNT };
    enum waveforms { square, sine, triangle };
    enum methods { ExplicitEuler, SemiExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
}

__global__ void kernelProgram_(name)(Computation* data)
{
    uint64_t variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
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

__device__ __forceinline__  void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters)
{
    ifSIGNAL(P(signal), square)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmodf(V(theta) + H * V(v), 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));
        }

        ifMETHOD(P(method), SemiExplicitEuler)
        {
            Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmodf(V(theta) + H * V(v), 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(Vnext(theta))) * V(v) - sinf(Vnext(theta)) - P(delta) * Vnext(iL)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            numb tmp = V(t) + H * 0.5;

            numb thetamp = fmodf(V(theta) + H * 0.5 * V(v), 2.0f * 3.141592f);
            numb iLmp = V(iL) + H * 0.5 * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + H * 0.5 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));

            Vnext(i) = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(t) = V(t) + H;

            Vnext(theta) = fmodf(V(theta) + H * vmp, 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (vmp - iLmp));
            Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb i1 = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            numb kt1 = V(t) + 0.5f * H;

            numb ktheta1 = V(v);
            numb kiL1 = ((1.0f / P(betaL)) * (V(v) - V(iL)));
            numb kv1 = ((1.0f / P(betaC)) * (i1 - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));

            numb thetamp = fmodf(V(theta) + H * 0.5 * ktheta1, 2.0f * 3.141592f);
            numb iLmp = V(iL) + H * 0.5 * kiL1;
            numb vmp = V(v) + H * 0.5 * kv1;

            numb i2 = P(Idc) + (fmodf((kt1 - P(Idel)) > 0 ? (kt1 - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - kt1), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

            numb ktheta2 = vmp;
            numb kiL2 = ((1.0f / P(betaL)) * (vmp - iLmp));
            numb kv2 = ((1.0f / P(betaC)) * (i2 - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

            thetamp = fmodf(V(theta) + H * 0.5 * ktheta2, 2.0f * 3.141592f);
            iLmp = V(iL) + H * 0.5 * kiL2;
            vmp = V(v) + H * 0.5 * kv2;

            numb i3 = P(Idc) + (fmodf((kt1 - P(Idel)) > 0 ? (kt1 - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - kt1), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(t) = V(t) + H;

            numb ktheta3 = vmp;
            numb kiL3 = ((1.0f / P(betaL)) * (vmp - iLmp));
            numb kv3 = ((1.0f / P(betaC)) * (i3 - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

            thetamp = fmodf(V(theta) + H * ktheta3, 2.0f * 3.141592f);
            iLmp = V(iL) + H * kiL3;
            vmp = V(v) + H * kv3;

            numb i4 = P(Idc) + (fmodf((Vnext(t) - P(Idel)) > 0 ? (Vnext(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - Vnext(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

            numb ktheta4 = vmp;
            numb kiL4 = ((1.0f / P(betaL)) * (vmp - iLmp));
            numb kv4 = ((1.0f / P(betaC)) * (i4 - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

            Vnext(i) = i4;
            Vnext(theta) = fmodf(V(theta) + H * (ktheta1 + 2 * ktheta2 + 2 * ktheta3 + ktheta4) / 6, 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * (kiL1 + 2 * kiL2 + 2 * kiL3 + kiL4) / 6;
            Vnext(v) = V(v) + H * (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = 0.5 * H - P(symmetry);
            numb h2 = 0.5 * H + P(symmetry);

            Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(t) = V(t) + h1;
            numb thetamp = fmodf(V(theta) + h1 * V(v), 2.0f * 3.141592f);
            numb iLmp = V(iL) + h1 * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + h1 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * V(v) - sinf(thetamp) - P(delta) * iLmp));

            Vnext(t) = V(t) + h2;
            Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(v) = vmp + h2 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));
            Vnext(v) = vmp + h2 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * Vnext(v) - sinf(thetamp) - P(delta) * iLmp));
            Vnext(iL) = (iLmp + h2 * (1.0f / P(betaL)) * Vnext(v)) / (1 + h2 * (1.0f / P(betaL)));
            Vnext(theta) = fmodf(thetamp + h2 * Vnext(v), 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
        }
    }
    ifSIGNAL(P(signal), sine)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmodf(V(theta) + H * V(v), 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));
        }

        ifMETHOD(P(method), SemiExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmodf(V(theta) + H * V(v), 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(Vnext(theta))) * V(v) - sinf(Vnext(theta)) - P(delta) * Vnext(iL)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
            numb tmp = V(t) + H * 0.5;

            numb thetamp = fmodf(V(theta) + H * 0.5 * V(v), 2.0f * 3.141592f);
            numb iLmp = V(iL) + H * 0.5 * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + H * 0.5 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));

            Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + H;

            Vnext(theta) = fmodf(V(theta) + H * vmp, 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (vmp - iLmp));
            Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb i1 = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
            numb kt1 = V(t) + 0.5f * H;

            numb ktheta1 = V(v);
            numb kiL1 = ((1.0f / P(betaL)) * (V(v) - V(iL)));
            numb kv1 = ((1.0f / P(betaC)) * (i1 - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));

            numb thetamp = fmodf(V(theta) + H * 0.5 * ktheta1, 2.0f * 3.141592f);
            numb iLmp = V(iL) + H * 0.5 * kiL1;
            numb vmp = V(v) + H * 0.5 * kv1;

            numb i2 = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));

            numb ktheta2 = vmp;
            numb kiL2 = ((1.0f / P(betaL)) * (vmp - iLmp));
            numb kv2 = ((1.0f / P(betaC)) * (i2 - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

            thetamp = fmodf(V(theta) + H * 0.5 * ktheta2, 2.0f * 3.141592f);
            iLmp = V(iL) + H * 0.5 * kiL2;
            vmp = V(v) + H * 0.5 * kv2;

            numb i3 = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + H;

            numb ktheta3 = vmp;
            numb kiL3 = ((1.0f / P(betaL)) * (vmp - iLmp));
            numb kv3 = ((1.0f / P(betaC)) * (i3 - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

            thetamp = fmodf(V(theta) + H * ktheta3, 2.0f * 3.141592f);
            iLmp = V(iL) + H * kiL3;
            vmp = V(v) + H * kv3;

            numb i4 = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));

            numb ktheta4 = vmp;
            numb kiL4 = ((1.0f / P(betaL)) * (vmp - iLmp));
            numb kv4 = ((1.0f / P(betaC)) * (i4 - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

            Vnext(i) = i4;
            Vnext(theta) = fmodf(V(theta) + H * (ktheta1 + 2 * ktheta2 + 2 * ktheta3 + ktheta4) / 6, 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * (kiL1 + 2 * kiL2 + 2 * kiL3 + kiL4) / 6;
            Vnext(v) = V(v) + H * (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = 0.5 * H - P(symmetry);
            numb h2 = 0.5 * H + P(symmetry);

            Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + h1;
            numb thetamp = fmodf(V(theta) + h1 * V(v), 2.0f * 3.141592f);
            numb iLmp = V(iL) + h1 * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + h1 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * V(v) - sinf(thetamp) - P(delta) * iLmp));

            Vnext(t) = V(t) + h2;
            Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(v) = vmp + h2 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));
            Vnext(v) = vmp + h2 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * Vnext(v) - sinf(thetamp) - P(delta) * iLmp));
            Vnext(iL) = (iLmp + h2 * (1.0f / P(betaL)) * Vnext(v)) / (1 + h2 * (1.0f / P(betaL)));
            Vnext(theta) = fmodf(thetamp + h2 * Vnext(v), 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
        }
    }
    ifSIGNAL(P(signal), triangle)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmodf(V(theta) + H * V(v), 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));
        }

        ifMETHOD(P(method), SemiExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmodf(V(theta) + H * V(v), 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(Vnext(theta))) * V(v) - sinf(Vnext(theta)) - P(delta) * Vnext(iL)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
            numb tmp = V(t) + H * 0.5;

            numb thetamp = fmodf(V(theta) + H * 0.5 * V(v), 2.0f * 3.141592f);
            numb iLmp = V(iL) + H * 0.5 * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + H * 0.5 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));

            Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
            Vnext(t) = V(t) + H;

            Vnext(theta) = fmodf(V(theta) + H * vmp, 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * ((1.0f / P(betaL)) * (vmp - iLmp));
            Vnext(v) = V(v) + H * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb i1 = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
            numb kt1 = V(t) + 0.5f * H;

            numb ktheta1 = V(v);
            numb kiL1 = ((1.0f / P(betaL)) * (V(v) - V(iL)));
            numb kv1 = ((1.0f / P(betaC)) * (i1 - P(betaM) * (1 + P(epsilon) * cosf(V(theta))) * V(v) - sinf(V(theta)) - P(delta) * V(iL)));

            numb thetamp = fmodf(V(theta) + H * 0.5 * ktheta1, 2.0f * 3.141592f);
            numb iLmp = V(iL) + H * 0.5 * kiL1;
            numb vmp = V(v) + H * 0.5 * kv1;

            numb i2 = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));

            numb ktheta2 = vmp;
            numb kiL2 = ((1.0f / P(betaL)) * (vmp - iLmp));
            numb kv2 = ((1.0f / P(betaC)) * (i2 - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

            thetamp = fmodf(V(theta) + H * 0.5 * ktheta2, 2.0f * 3.141592f);
            iLmp = V(iL) + H * 0.5 * kiL2;
            vmp = V(v) + H * 0.5 * kv2;

            numb i3 = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
            Vnext(t) = V(t) + H;

            numb ktheta3 = vmp;
            numb kiL3 = ((1.0f / P(betaL)) * (vmp - iLmp));
            numb kv3 = ((1.0f / P(betaC)) * (i3 - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

            thetamp = fmodf(V(theta) + H * ktheta3, 2.0f * 3.141592f);
            iLmp = V(iL) + H * kiL3;
            vmp = V(v) + H * kv3;

            numb i4 = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));

            numb ktheta4 = vmp;
            numb kiL4 = ((1.0f / P(betaL)) * (vmp - iLmp));
            numb kv4 = ((1.0f / P(betaC)) * (i4 - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));

            Vnext(i) = i4;
            Vnext(theta) = fmodf(V(theta) + H * (ktheta1 + 2 * ktheta2 + 2 * ktheta3 + ktheta4) / 6, 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
            Vnext(iL) = V(iL) + H * (kiL1 + 2 * kiL2 + 2 * kiL3 + kiL4) / 6;
            Vnext(v) = V(v) + H * (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = 0.5 * H - P(symmetry);
            numb h2 = 0.5 * H + P(symmetry);

            Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
            Vnext(t) = V(t) + h1;
            numb thetamp = fmodf(V(theta) + h1 * V(v), 2.0f * 3.141592f);
            numb iLmp = V(iL) + h1 * ((1.0f / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + h1 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * V(v) - sinf(thetamp) - P(delta) * iLmp));

            Vnext(t) = V(t) + h2;
            Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
            Vnext(v) = vmp + h2 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * vmp - sinf(thetamp) - P(delta) * iLmp));
            Vnext(v) = vmp + h2 * ((1.0f / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cosf(thetamp)) * Vnext(v) - sinf(thetamp) - P(delta) * iLmp));
            Vnext(iL) = (iLmp + h2 * (1.0f / P(betaL)) * Vnext(v)) / (1 + h2 * (1.0f / P(betaL)));
            Vnext(theta) = fmodf(thetamp + h2 * Vnext(v), 2.0f * 3.141592f);
            Vnext(sin_theta) = sinf(Vnext(theta));
        }
    }
}

#undef name