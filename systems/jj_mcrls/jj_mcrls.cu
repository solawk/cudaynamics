#include "jj_mcrls.h"

#define name jj_mcrls

namespace attributes
{
    enum variables { theta, sin_theta, v, iL, i, t };
    enum parameters { betaL, betaC, betaM, epsilon, delta, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, method, COUNT };
    enum waveforms { square, sine, triangle };
    enum methods { ExplicitEuler, SemiExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
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
    ifSIGNAL(P(signal), square)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmod(V(theta) + H * V(v), (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(V(theta))) * V(v) - sin(V(theta)) - P(delta) * V(iL)));
        }

        ifMETHOD(P(method), SemiExplicitEuler)
        {
            Vnext(i) = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmod(V(theta) + H * V(v), (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(Vnext(theta))) * V(v) - sin(Vnext(theta)) - P(delta) * Vnext(iL)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb tmp = V(t) + H * (numb)0.5;

            numb thetamp = fmod(V(theta) + H * (numb)0.5 * V(v), (numb)2.0 * (numb)3.141592);
            numb iLmp = V(iL) + H * (numb)0.5 * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + H * (numb)0.5 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(V(theta))) * V(v) - sin(V(theta)) - P(delta) * V(iL)));

            Vnext(i) = P(Idc) + (fmod((tmp - P(Idel)) > (numb)0.0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(t) = V(t) + H;

            Vnext(theta) = fmod(V(theta) + H * vmp, (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            Vnext(v) = V(v) + H * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb i1 = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb kt1 = V(t) + (numb)0.5 * H;

            numb ktheta1 = V(v);
            numb kiL1 = (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            numb kv1 = (((numb)1.0 / P(betaC)) * (i1 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(V(theta))) * V(v) - sin(V(theta)) - P(delta) * V(iL)));

            numb thetamp = fmod(V(theta) + H * (numb)0.5 * ktheta1, (numb)2.0 * (numb)3.141592);
            numb iLmp = V(iL) + H * (numb)0.5 * kiL1;
            numb vmp = V(v) + H * (numb)0.5 * kv1;

            numb i2 = P(Idc) + (fmod((kt1 - P(Idel)) > (numb)0.0 ? (kt1 - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - kt1), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);

            numb ktheta2 = vmp;
            numb kiL2 = (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            numb kv2 = (((numb)1.0 / P(betaC)) * (i2 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));

            thetamp = fmod(V(theta) + H * (numb)0.5 * ktheta2, (numb)2.0 * (numb)3.141592);
            iLmp = V(iL) + H * (numb)0.5 * kiL2;
            vmp = V(v) + H * (numb)0.5 * kv2;

            numb i3 = P(Idc) + (fmod((kt1 - P(Idel)) > (numb)0.0 ? (kt1 - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - kt1), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(t) = V(t) + H;

            numb ktheta3 = vmp;
            numb kiL3 = (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            numb kv3 = (((numb)1.0 / P(betaC)) * (i3 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));

            thetamp = fmod(V(theta) + H * ktheta3, (numb)2.0 * (numb)3.141592);
            iLmp = V(iL) + H * kiL3;
            vmp = V(v) + H * kv3;

            numb i4 = P(Idc) + (fmod((Vnext(t) - P(Idel)) > (numb)0.0 ? (Vnext(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - Vnext(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);

            numb ktheta4 = vmp;
            numb kiL4 = (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            numb kv4 = (((numb)1.0 / P(betaC)) * (i4 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));

            Vnext(i) = i4;
            Vnext(theta) = fmod(V(theta) + H * (ktheta1 + (numb)2.0 * ktheta2 + (numb)2.0 * ktheta3 + ktheta4) / (numb)6.0, (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (kiL1 + (numb)2.0 * kiL2 + (numb)2.0 * kiL3 + kiL4) / (numb)6.0;
            Vnext(v) = V(v) + H * (kv1 + (numb)2.0 * kv2 + (numb)2.0 * kv3 + kv4) / (numb)6.0;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = (numb)0.5 * H - P(symmetry);
            numb h2 = (numb)0.5 * H + P(symmetry);

            Vnext(i) = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(t) = V(t) + h1;
            numb thetamp = fmod(V(theta) + h1 * V(v), (numb)2.0 * (numb)3.141592);
            numb iLmp = V(iL) + h1 * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + h1 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * V(v) - sin(thetamp) - P(delta) * iLmp));

            Vnext(t) = V(t) + h2;
            Vnext(i) = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(v) = vmp + h2 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));
            Vnext(v) = vmp + h2 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * Vnext(v) - sin(thetamp) - P(delta) * iLmp));
            Vnext(iL) = (iLmp + h2 * ((numb)1.0 / P(betaL)) * Vnext(v)) / ((numb)1.0 + h2 * ((numb)1.0 / P(betaL)));
            Vnext(theta) = fmod(thetamp + h2 * Vnext(v), (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
        }
    }
    ifSIGNAL(P(signal), sine)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmod(V(theta) + H * V(v), (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(V(theta))) * V(v) - sin(V(theta)) - P(delta) * V(iL)));
        }

        ifMETHOD(P(method), SemiExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmod(V(theta) + H * V(v), (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(Vnext(theta))) * V(v) - sin(Vnext(theta)) - P(delta) * Vnext(iL)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            numb tmp = V(t) + H * (numb)0.5;

            numb thetamp = fmod(V(theta) + H * (numb)0.5 * V(v), (numb)2.0 * (numb)3.141592);
            numb iLmp = V(iL) + H * (numb)0.5 * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + H * (numb)0.5 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(V(theta))) * V(v) - sin(V(theta)) - P(delta) * V(iL)));

            Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + H;

            Vnext(theta) = fmod(V(theta) + H * vmp, (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            Vnext(v) = V(v) + H * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb i1 = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            numb kt1 = V(t) + (numb)0.5 * H;

            numb ktheta1 = V(v);
            numb kiL1 = (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            numb kv1 = (((numb)1.0 / P(betaC)) * (i1 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(V(theta))) * V(v) - sin(V(theta)) - P(delta) * V(iL)));

            numb thetamp = fmod(V(theta) + H * (numb)0.5 * ktheta1, (numb)2.0 * (numb)3.141592);
            numb iLmp = V(iL) + H * (numb)0.5 * kiL1;
            numb vmp = V(v) + H * (numb)0.5 * kv1;

            numb i2 = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));

            numb ktheta2 = vmp;
            numb kiL2 = (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            numb kv2 = (((numb)1.0 / P(betaC)) * (i2 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));

            thetamp = fmod(V(theta) + H * (numb)0.5 * ktheta2, (numb)2.0 * (numb)3.141592);
            iLmp = V(iL) + H * (numb)0.5 * kiL2;
            vmp = V(v) + H * (numb)0.5 * kv2;

            numb i3 = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + H;

            numb ktheta3 = vmp;
            numb kiL3 = (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            numb kv3 = (((numb)1.0 / P(betaC)) * (i3 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));

            thetamp = fmod(V(theta) + H * ktheta3, (numb)2.0 * (numb)3.141592);
            iLmp = V(iL) + H * kiL3;
            vmp = V(v) + H * kv3;

            numb i4 = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));

            numb ktheta4 = vmp;
            numb kiL4 = (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            numb kv4 = (((numb)1.0 / P(betaC)) * (i4 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));

            Vnext(i) = i4;
            Vnext(theta) = fmod(V(theta) + H * (ktheta1 + (numb)2.0 * ktheta2 + (numb)2.0 * ktheta3 + ktheta4) / (numb)6.0, (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (kiL1 + (numb)2.0 * kiL2 + (numb)2.0 * kiL3 + kiL4) / (numb)6.0;
            Vnext(v) = V(v) + H * (kv1 + (numb)2.0 * kv2 + (numb)2.0 * kv3 + kv4) / (numb)6.0;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = (numb)0.5 * H - P(symmetry);
            numb h2 = (numb)0.5 * H + P(symmetry);

            Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + h1;
            numb thetamp = fmod(V(theta) + h1 * V(v), (numb)2.0 * (numb)3.141592);
            numb iLmp = V(iL) + h1 * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + h1 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * V(v) - sin(thetamp) - P(delta) * iLmp));

            Vnext(t) = V(t) + h2;
            Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(v) = vmp + h2 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));
            Vnext(v) = vmp + h2 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * Vnext(v) - sin(thetamp) - P(delta) * iLmp));
            Vnext(iL) = (iLmp + h2 * ((numb)1.0 / P(betaL)) * Vnext(v)) / ((numb)1.0 + h2 * ((numb)1.0 / P(betaL)));
            Vnext(theta) = fmod(thetamp + h2 * Vnext(v), (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
        }
    }
    ifSIGNAL(P(signal), triangle)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)) * pow((-1), floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)));
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmod(V(theta) + H * V(v), (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * (1 + P(epsilon) * cos(V(theta))) * V(v) - sin(V(theta)) - P(delta) * V(iL)));
        }

        ifMETHOD(P(method), SemiExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
            Vnext(t) = V(t) + H;
            Vnext(theta) = fmod(V(theta) + H * V(v), (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            Vnext(v) = V(v) + H * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(Vnext(theta))) * V(v) - sin(Vnext(theta)) - P(delta) * Vnext(iL)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
            numb tmp = V(t) + H * (numb)0.5;

            numb thetamp = fmod(V(theta) + H * (numb)0.5 * V(v), (numb)2.0 * (numb)3.141592);
            numb iLmp = V(iL) + H * (numb)0.5 * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + H * (numb)0.5 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(V(theta))) * V(v) - sin(V(theta)) - P(delta) * V(iL)));

            Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
            Vnext(t) = V(t) + H;

            Vnext(theta) = fmod(V(theta) + H * vmp, (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            Vnext(v) = V(v) + H * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb i1 = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
            numb kt1 = V(t) + (numb)0.5 * H;

            numb ktheta1 = V(v);
            numb kiL1 = (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            numb kv1 = (((numb)1.0 / P(betaC)) * (i1 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(V(theta))) * V(v) - sin(V(theta)) - P(delta) * V(iL)));

            numb thetamp = fmod(V(theta) + H * (numb)0.5 * ktheta1, (numb)2.0 * (numb)3.141592);
            numb iLmp = V(iL) + H * (numb)0.5 * kiL1;
            numb vmp = V(v) + H * (numb)0.5 * kv1;

            numb i2 = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));

            numb ktheta2 = vmp;
            numb kiL2 = (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            numb kv2 = (((numb)1.0 / P(betaC)) * (i2 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));

            thetamp = fmod(V(theta) + H * (numb)0.5 * ktheta2, (numb)2.0 * (numb)3.141592);
            iLmp = V(iL) + H * (numb)0.5 * kiL2;
            vmp = V(v) + H * (numb)0.5 * kv2;

            numb i3 = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
            Vnext(t) = V(t) + H;

            numb ktheta3 = vmp;
            numb kiL3 = (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            numb kv3 = (((numb)1.0 / P(betaC)) * (i3 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));

            thetamp = fmod(V(theta) + H * ktheta3, (numb)2.0 * (numb)3.141592);
            iLmp = V(iL) + H * kiL3;
            vmp = V(v) + H * kv3;

            numb i4 = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));

            numb ktheta4 = vmp;
            numb kiL4 = (((numb)1.0 / P(betaL)) * (vmp - iLmp));
            numb kv4 = (((numb)1.0 / P(betaC)) * (i4 - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));

            Vnext(i) = i4;
            Vnext(theta) = fmod(V(theta) + H * (ktheta1 + (numb)2.0 * ktheta2 + (numb)2.0 * ktheta3 + ktheta4) / (numb)6.0, (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
            Vnext(iL) = V(iL) + H * (kiL1 + (numb)2.0 * kiL2 + (numb)2.0 * kiL3 + kiL4) / (numb)6.0;
            Vnext(v) = V(v) + H * (kv1 + (numb)2.0 * kv2 + (numb)2.0 * kv3 + kv4) / (numb)6.0;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = (numb)0.5 * H - P(symmetry);
            numb h2 = (numb)0.5 * H + P(symmetry);

            Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
            Vnext(t) = V(t) + h1;
            numb thetamp = fmod(V(theta) + h1 * V(v), (numb)2.0 * (numb)3.141592);
            numb iLmp = V(iL) + h1 * (((numb)1.0 / P(betaL)) * (V(v) - V(iL)));
            numb vmp = V(v) + h1 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * V(v) - sin(thetamp) - P(delta) * iLmp));

            Vnext(t) = V(t) + h2;
            Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
            Vnext(v) = vmp + h2 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * vmp - sin(thetamp) - P(delta) * iLmp));
            Vnext(v) = vmp + h2 * (((numb)1.0 / P(betaC)) * (Vnext(i) - P(betaM) * ((numb)1.0 + P(epsilon) * cos(thetamp)) * Vnext(v) - sin(thetamp) - P(delta) * iLmp));
            Vnext(iL) = (iLmp + h2 * ((numb)1.0 / P(betaL)) * Vnext(v)) / ((numb)1.0 + h2 * ((numb)1.0 / P(betaL)));
            Vnext(theta) = fmod(thetamp + h2 * Vnext(v), (numb)2.0 * (numb)3.141592);
            Vnext(sin_theta) = sin(Vnext(theta));
        }
    }
}

#undef name