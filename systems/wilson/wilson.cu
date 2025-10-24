#include "main.h"
#include "wilson.h"

namespace attributes
{
    enum variables { v, r, i, t };
    enum parameters { C, tau, p0, p1, p2, p3, p4, p5, p6, p7, Idc, Iamp, Ifreq, Idel, Idf, signal, method, COUNT };
    enum waveforms { square, sine, triangle };
    enum methods { ExplicitEuler, ExplicitMidpoint };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_wilson(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_wilson);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_wilson(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_wilson, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_wilson, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBscan_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4),
            H);
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_wilson, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__ void finiteDifferenceScheme_wilson(numb* currentV, numb* nextV, numb* parameters)
{
    ifSIGNAL(P(signal), square)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(t) = V(t) + H;
            Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
            Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
        }
        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            numb tmp = V(t) + H * 0.5;
            numb vmp = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
            numb rmp = V(r) + H * 0.5 * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

            Vnext(i) = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(t) = V(t) + H;
            Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + Vnext(i)) / P(C));
            Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7)));
        }
    }
    ifSIGNAL(P(signal), sine)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + H;
            Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
            Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
        }
        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
            numb tmp = V(t) + H * 0.5;
            numb vmp = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
            numb rmp = V(r) + H * 0.5 * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

            Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));
            Vnext(t) = V(t) + H;
            Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + Vnext(i)) / P(C));
            Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7)));
        }
    }
    ifSIGNAL(P(signal), triangle)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * ((4 * P(Ifreq) * (V(t) - P(Idel)) - 2 * floorf((4 * P(Ifreq) * (V(t) - P(Idel)) + 1) / 2)) * pow((-1), floorf((4 * P(Ifreq) * (V(t) - P(Idel)) + 1) / 2)));
            Vnext(t) = V(t) + H;
            Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
            Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
        }
        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + P(Iamp) * ((4 * P(Ifreq) * (V(t) - P(Idel)) - 2 * floorf((4 * P(Ifreq) * (V(t) - P(Idel)) + 1) / 2)) * pow((-1), floorf((4 * P(Ifreq) * (V(t) - P(Idel)) + 1) / 2)));
            numb tmp = V(t) + H * 0.5;
            numb vmp = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
            numb rmp = V(r) + H * 0.5 * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

            Vnext(i) = P(Idc) + P(Iamp) * ((4 * P(Ifreq) * (tmp - P(Idel)) - 2 * floorf((4 * P(Ifreq) * (tmp - P(Idel)) + 1) / 2)) * pow((-1), floorf((4 * P(Ifreq) * (tmp - P(Idel)) + 1) / 2)));
            Vnext(t) = V(t) + H;
            Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + Vnext(i)) / P(C));
            Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7)));
        }
    }
}