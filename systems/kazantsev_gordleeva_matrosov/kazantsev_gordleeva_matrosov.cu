#include "kazantsev_gordleeva_matrosov.h"

#define name kazantsev_gordleeva_matrosov

namespace attributes
{
    enum variables { p, q, z, j, t };
    enum parameters { c_IP3_e, tau_IP3, v4, alpha, k4, c1, v1, c0, d1, d5, v3, k3, v2, v5, v6, k2, k1, a2, d2, d3, Jdc, Jamp, Jfreq, Jdel, Jdf, symmetry, signal, method, COUNT };
    enum waveforms { square, sine, triangle };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
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
            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) / (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            Vnext(j) = P(Jdc) + (fmodf((V(t) - P(Jdel)) > 0 ? (V(t) - P(Jdel)) : (P(Jdf) / P(Jfreq) + P(Jdel) - V(t)), 1 / P(Jfreq)) < P(Jdf) / P(Jfreq) ? P(Jamp) : 0.0f);
            Vnext(t) = V(t) + H;

            Vnext(p) = V(p) + H * ((P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + Vnext(j));
            Vnext(q) = V(q) + H * (j_ch - j_pump + j_leak + j_in - j_out);
            Vnext(z) = V(z) + H * (P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) / (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            numb jmp = P(Jdc) + (fmodf((V(t) - P(Jdel)) > 0 ? (V(t) - P(Jdel)) : (P(Jdf) / P(Jfreq) + P(Jdel) - V(t)), 1 / P(Jfreq)) < P(Jdf) / P(Jfreq) ? P(Jamp) : 0.0f);
            numb tmp = V(t) + H * 0.5f;

            numb kp1 = (P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + jmp;
            numb kq1 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz1 = P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q));

            numb pmp = V(p) + 0.5f * H * kp1;
            numb qmp = V(q) + 0.5f * H * kq1;
            numb zmp = V(z) + 0.5f * H * kz1;

            Vnext(j) = P(Jdc) + (fmodf((tmp - P(Jdel)) > 0 ? (tmp - P(Jdel)) : (P(Jdf) / P(Jfreq) + P(Jdel) - tmp), 1 / P(Jfreq)) < P(Jdf) / P(Jfreq) ? P(Jamp) : 0.0f);
            Vnext(t) = V(t) + H;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            numb kp2 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp;
            numb kq2 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz2 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);
            
            Vnext(p) = V(p) + H * kp2;
            Vnext(q) = V(q) + H * kq2;
            Vnext(z) = V(z) + H * kz2;
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) / (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            numb jmp = P(Jdc) + (fmodf((V(t) - P(Jdel)) > 0 ? (V(t) - P(Jdel)) : (P(Jdf) / P(Jfreq) + P(Jdel) - V(t)), 1 / P(Jfreq)) < P(Jdf) / P(Jfreq) ? P(Jamp) : 0.0f);
            numb tmp = V(t) + H * 0.5f;

            numb kp1 = (P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + jmp;
            numb kq1 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz1 = P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q));

            numb pmp = V(p) + 0.5f * H * kp1;
            numb qmp = V(q) + 0.5f * H * kq1;
            numb zmp = V(z) + 0.5f * H * kz1;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            jmp = P(Jdc) + (fmodf((V(t) - P(Jdel)) > 0 ? (tmp - P(Jdel)) : (P(Jdf) / P(Jfreq) + P(Jdel) - tmp), 1 / P(Jfreq)) < P(Jdf) / P(Jfreq) ? P(Jamp) : 0.0f);
            tmp = V(t) + H * 0.5f;

            numb kp2 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp;
            numb kq2 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz2 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);

            pmp = V(p) + 0.5f * H * kp2;
            qmp = V(q) + 0.5f * H * kq2;
            zmp = V(z) + 0.5f * H * kz2;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            jmp = P(Jdc) + (fmodf((V(t) - P(Jdel)) > 0 ? (tmp - P(Jdel)) : (P(Jdf) / P(Jfreq) + P(Jdel) - tmp), 1 / P(Jfreq)) < P(Jdf) / P(Jfreq) ? P(Jamp) : 0.0f);
            tmp = V(t) + H;

            numb kp3 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp;
            numb kq3 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz3 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);

            pmp = V(p) + H * kp3;
            qmp = V(q) + H * kq3;
            zmp = V(z) + H * kz3;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            Vnext(j) = P(Jdc) + (fmodf((V(t) - P(Jdel)) > 0 ? (tmp - P(Jdel)) : (P(Jdf) / P(Jfreq) + P(Jdel) - tmp), 1 / P(Jfreq)) < P(Jdf) / P(Jfreq) ? P(Jamp) : 0.0f);
            Vnext(t) = V(t) + H;

            numb kp4 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + Vnext(j);
            numb kq4 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz4 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);

            Vnext(p) = V(p) + H * (kp1 + 2.0f * kp2 + 2.0f * kp3 + kp4) / 6.0f;
            Vnext(q) = V(q) + H * (kq1 + 2.0f * kq2 + 2.0f * kq3 + kq4) / 6.0f;
            Vnext(z) = V(z) + H * (kz1 + 2.0f * kz2 + 2.0f * kz3 + kz4) / 6.0f;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = 0.5 * H - P(symmetry);
            numb h2 = 0.5 * H + P(symmetry);

            numb tmp = V(t) + h1;
            numb jmp = P(Jdc) + (fmodf((tmp - P(Jdel)) > 0 ? (tmp - P(Jdel)) :
                (P(Jdf) / P(Jfreq) + P(Jdel) - tmp),
                1 / P(Jfreq)) < P(Jdf) / P(Jfreq) ? P(Jamp) : 0.0f);

            Vnext(j) = jmp;
            Vnext(t) = V(t) + H;

            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) /
                (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            numb pmp = V(p) + h1 * ((P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + jmp);
            numb qmp = V(q) + h1 * (j_ch - j_pump + j_leak + j_in - j_out);
            numb zmp = V(z) + h1 * (P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q)));

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) /
                (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            Vnext(p) = pmp + h2 * ((P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp);
            Vnext(q) = qmp + h2 * (j_ch - j_pump + j_leak + j_in - j_out);
            Vnext(z) = zmp + h2 * (P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp));
        }

    }

    ifSIGNAL(P(signal), sine)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) / (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            Vnext(j) = P(Jdc) + P(Jamp) * sinf(2.0f * 3.141592f * P(Jfreq) * (V(t) - P(Jdel)));
            Vnext(t) = V(t) + H;

            Vnext(p) = V(p) + H * ((P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + Vnext(j));
            Vnext(q) = V(q) + H * (j_ch - j_pump + j_leak + j_in - j_out);
            Vnext(z) = V(z) + H * (P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) / (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            numb jmp = P(Jdc) + P(Jamp) * sinf(2.0f * 3.141592f * P(Jfreq) * (V(t) - P(Jdel)));
            numb tmp = V(t) + H * 0.5f;

            numb kp1 = (P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + jmp;
            numb kq1 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz1 = P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q));

            numb pmp = V(p) + 0.5f * H * kp1;
            numb qmp = V(q) + 0.5f * H * kq1;
            numb zmp = V(z) + 0.5f * H * kz1;

            Vnext(j) = P(Jdc) + P(Jamp) * sinf(2.0f * 3.141592f * P(Jfreq) * (tmp - P(Jdel)));
            Vnext(t) = V(t) + H;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            numb kp2 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp;
            numb kq2 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz2 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);

            Vnext(p) = V(p) + H * kp2;
            Vnext(q) = V(q) + H * kq2;
            Vnext(z) = V(z) + H * kz2;
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) / (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            numb jmp = P(Jdc) + P(Jamp) * sinf(2.0f * 3.141592f * P(Jfreq) * (V(t) - P(Jdel)));
            numb tmp = V(t) + H * 0.5f;

            numb kp1 = (P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + jmp;
            numb kq1 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz1 = P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q));

            numb pmp = V(p) + 0.5f * H * kp1;
            numb qmp = V(q) + 0.5f * H * kq1;
            numb zmp = V(z) + 0.5f * H * kz1;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            jmp = P(Jdc) + P(Jamp) * sinf(2.0f * 3.141592f * P(Jfreq) * (tmp - P(Jdel)));
            tmp = V(t) + H * 0.5f;

            numb kp2 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp;
            numb kq2 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz2 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);

            pmp = V(p) + 0.5f * H * kp2;
            qmp = V(q) + 0.5f * H * kq2;
            zmp = V(z) + 0.5f * H * kz2;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            jmp = P(Jdc) + P(Jamp) * sinf(2.0f * 3.141592f * P(Jfreq) * (tmp - P(Jdel)));
            tmp = V(t) + H;

            numb kp3 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp;
            numb kq3 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz3 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);

            pmp = V(p) + H * kp3;
            qmp = V(q) + H * kq3;
            zmp = V(z) + H * kz3;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            Vnext(j) = P(Jdc) + P(Jamp) * sinf(2.0f * 3.141592f * P(Jfreq) * (tmp - P(Jdel)));
            Vnext(t) = V(t) + H;

            numb kp4 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + Vnext(j);
            numb kq4 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz4 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);

            Vnext(p) = V(p) + H * (kp1 + 2.0f * kp2 + 2.0f * kp3 + kp4) / 6.0f;
            Vnext(q) = V(q) + H * (kq1 + 2.0f * kq2 + 2.0f * kq3 + kq4) / 6.0f;
            Vnext(z) = V(z) + H * (kz1 + 2.0f * kz2 + 2.0f * kz3 + kz4) / 6.0f;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = 0.5 * H - P(symmetry);
            numb h2 = 0.5 * H + P(symmetry);

            numb tmp = V(t) + h1;
            numb jmp = P(Jdc) + P(Jamp) * sinf(2.0f * 3.141592f * P(Jfreq) * (V(t) - P(Jdel)));

            Vnext(j) = jmp;
            Vnext(t) = V(t) + H;

            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) /
                (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            numb pmp = V(p) + h1 * ((P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + jmp);
            numb qmp = V(q) + h1 * (j_ch - j_pump + j_leak + j_in - j_out);
            numb zmp = V(z) + h1 * (P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q)));

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) /
                (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            Vnext(p) = pmp + h2 * ((P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp);
            Vnext(q) = qmp + h2 * (j_ch - j_pump + j_leak + j_in - j_out);
            Vnext(z) = zmp + h2 * (P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp));
        }
    }

    ifSIGNAL(P(signal), triangle)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) / (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            Vnext(j) = P(Jdc) + P(Jamp) * ((4.0f * P(Jfreq) * (V(t) - P(Jdel)) - 2.0f * floorf((4.0f * P(Jfreq) * (V(t) - P(Jdel)) + 1.0f) / 
                2.0f)) * pow((-1), floorf((4.0f * P(Jfreq) * (V(t) - P(Jdel)) + 1.0f) / 2.0f)));
            Vnext(t) = V(t) + H;

            Vnext(p) = V(p) + H * ((P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + Vnext(j));
            Vnext(q) = V(q) + H * (j_ch - j_pump + j_leak + j_in - j_out);
            Vnext(z) = V(z) + H * (P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) / (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            numb jmp = P(Jdc) + P(Jamp) * ((4.0f * P(Jfreq) * (V(t) - P(Jdel)) - 2.0f * floorf((4.0f * P(Jfreq) * (V(t) - P(Jdel)) + 1.0f) /
                2.0f)) * pow((-1), floorf((4.0f * P(Jfreq) * (V(t) - P(Jdel)) + 1.0f) / 2.0f)));
            numb tmp = V(t) + H * 0.5f;

            numb kp1 = (P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + jmp;
            numb kq1 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz1 = P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q));

            numb pmp = V(p) + 0.5f * H * kp1;
            numb qmp = V(q) + 0.5f * H * kq1;
            numb zmp = V(z) + 0.5f * H * kz1;

            Vnext(j) = P(Jdc) + P(Jamp) * ((4.0f * P(Jfreq) * (tmp - P(Jdel)) - 2.0f * floorf((4.0f * P(Jfreq) * (tmp - P(Jdel)) + 1.0f) /
                2.0f)) * pow((-1), floorf((4.0f * P(Jfreq) * (tmp - P(Jdel)) + 1.0f) / 2.0f)));
            Vnext(t) = V(t) + H;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            numb kp2 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp;
            numb kq2 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz2 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);

            Vnext(p) = V(p) + H * kp2;
            Vnext(q) = V(q) + H * kq2;
            Vnext(z) = V(z) + H * kz2;
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) / (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            numb jmp = P(Jdc) + P(Jamp) * ((4.0f * P(Jfreq) * (V(t) - P(Jdel)) - 2.0f * floorf((4.0f * P(Jfreq) * (V(t) - P(Jdel)) + 1.0f) /
                2.0f)) * pow((-1), floorf((4.0f * P(Jfreq) * (V(t) - P(Jdel)) + 1.0f) / 2.0f)));
            numb tmp = V(t) + H * 0.5f;

            numb kp1 = (P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + jmp;
            numb kq1 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz1 = P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q));

            numb pmp = V(p) + 0.5f * H * kp1;
            numb qmp = V(q) + 0.5f * H * kq1;
            numb zmp = V(z) + 0.5f * H * kz1;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            jmp = P(Jdc) + P(Jamp) * ((4.0f * P(Jfreq) * (tmp - P(Jdel)) - 2.0f * floorf((4.0f * P(Jfreq) * (tmp - P(Jdel)) + 1.0f) /
                2.0f)) * pow((-1), floorf((4.0f * P(Jfreq) * (tmp - P(Jdel)) + 1.0f) / 2.0f)));
            tmp = V(t) + H * 0.5f;

            numb kp2 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp;
            numb kq2 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz2 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);

            pmp = V(p) + 0.5f * H * kp2;
            qmp = V(q) + 0.5f * H * kq2;
            zmp = V(z) + 0.5f * H * kz2;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            jmp = P(Jdc) + P(Jamp) * ((4.0f * P(Jfreq) * (tmp - P(Jdel)) - 2.0f * floorf((4.0f * P(Jfreq) * (tmp - P(Jdel)) + 1.0f) /
                2.0f)) * pow((-1), floorf((4.0f * P(Jfreq) * (tmp - P(Jdel)) + 1.0f) / 2.0f)));
            tmp = V(t) + H;

            numb kp3 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp;
            numb kq3 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz3 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);

            pmp = V(p) + H * kp3;
            qmp = V(q) + H * kq3;
            zmp = V(z) + H * kz3;

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) / (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            Vnext(j) = P(Jdc) + P(Jamp) * ((4.0f * P(Jfreq) * (tmp - P(Jdel)) - 2.0f * floorf((4.0f * P(Jfreq) * (tmp - P(Jdel)) + 1.0f) /
                2.0f)) * pow((-1), floorf((4.0f * P(Jfreq) * (tmp - P(Jdel)) + 1.0f) / 2.0f)));
            Vnext(t) = V(t) + H;

            numb kp4 = (P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + Vnext(j);
            numb kq4 = j_ch - j_pump + j_leak + j_in - j_out;
            numb kz4 = P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp);

            Vnext(p) = V(p) + H * (kp1 + 2.0f * kp2 + 2.0f * kp3 + kp4) / 6.0f;
            Vnext(q) = V(q) + H * (kq1 + 2.0f * kq2 + 2.0f * kq3 + kq4) / 6.0f;
            Vnext(z) = V(z) + H * (kz1 + 2.0f * kz2 + 2.0f * kz3 + kz4) / 6.0f;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = 0.5 * H - P(symmetry);
            numb h2 = 0.5 * H + P(symmetry);

            numb tmp = V(t) + h1;
            numb jmp = P(Jdc) + P(Jamp) * ((4.0f * P(Jfreq) * (V(t) - P(Jdel)) - 2.0f * floorf((4.0f * P(Jfreq) * (V(t) - P(Jdel)) + 1.0f) /
                2.0f)) * pow((-1), floorf((4.0f * P(Jfreq) * (V(t) - P(Jdel)) + 1.0f) / 2.0f)));

            Vnext(j) = jmp;
            Vnext(t) = V(t) + H;

            numb j_ch = (P(c1) * P(v1) * V(p) * V(p) * V(p) * V(z) * V(z) * V(z) * V(q) * V(q) * V(q) *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q))) /
                (((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))) * ((V(p) + P(d1)) * (V(q) + P(d5))));
            numb j_PLC = P(v4) * (V(q) + (1 - P(alpha)) * P(k4)) / (V(q) + P(k4));
            numb j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * V(q));
            numb j_pump = ((P(v3) * V(q) * V(q)) / (P(k3) * P(k3) + V(q) * V(q)));
            numb j_in = P(v5) + P(v6) * V(p) * V(p) / (P(k2) * P(k2) + V(p) * V(p));
            numb j_out = P(k1) * V(q);

            numb pmp = V(p) + h1 * ((P(c_IP3_e) - V(p)) / P(tau_IP3) + j_PLC + jmp);
            numb qmp = V(q) + h1 * (j_ch - j_pump + j_leak + j_in - j_out);
            numb zmp = V(z) + h1 * (P(a2) * (P(d2) * (V(p) + P(d1)) / (V(p) + P(d3)) * (1 - V(z)) - V(z) * V(q)));

            j_ch = (P(c1) * P(v1) * pmp * pmp * pmp * zmp * zmp * zmp * qmp * qmp * qmp *
                (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp)) /
                (((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))) * ((pmp + P(d1)) * (qmp + P(d5))));
            j_PLC = P(v4) * (qmp + (1 - P(alpha)) * P(k4)) / (qmp + P(k4));
            j_leak = P(c1) * P(v2) * (P(c0) / P(c1) - (1 + 1 / P(c1)) * qmp);
            j_pump = ((P(v3) * qmp * qmp) / (P(k3) * P(k3) + qmp * qmp));
            j_in = P(v5) + P(v6) * pmp * pmp / (P(k2) * P(k2) + pmp * pmp);
            j_out = P(k1) * qmp;

            Vnext(p) = pmp + h2 * ((P(c_IP3_e) - pmp) / P(tau_IP3) + j_PLC + jmp);
            Vnext(q) = qmp + h2 * (j_ch - j_pump + j_leak + j_in - j_out);
            Vnext(z) = zmp + h2 * (P(a2) * (P(d2) * (pmp + P(d1)) / (pmp + P(d3)) * (1 - zmp) - zmp * qmp));
        }
    }
}

#undef name