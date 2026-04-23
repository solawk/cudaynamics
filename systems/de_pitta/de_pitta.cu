#include "de_pitta.h"

#define name de_pitta

namespace attributes
{
    // ----------------------------
    // State variables (ODE states)
    // ----------------------------
    enum variables
    {
        GammaA,  // fraction of activated receptors (ΓA)
        I,       // IP3 (uM)
        C,       // cytosolic Ca2+ (uM)
        hGate,       // IP3R deinactivation gate
        YS,      // external neurotransmitter signal YS(t) (uM)
        t,
    };

    // ----------------------------
    // Parameters
    // ----------------------------
    enum parameters
    {
        // Metabotropic receptor kinetics
        O_N, Omega_N, zeta, K_KC,

        // IP3 signaling
        O_beta, O_delta, kappa_delta, K_delta,
        O_3K, K_D, K_3K,
        Omega_5P,

        // External IP3 drive (J_ex)
        F_ex, I_Theta, omega_I, I_bias,

        // Ca / IP3R
        Omega_C, Omega_L, O_P, K_P,
        C_T, rho_A,
        O_2,
        d1, d2, d3, d5,

        // YS
        rho_C, Y_T, Omega_C_YS, YSfreq, YSdel,

        // infrastructure
        symmetry, signal, method,

        COUNT
    };

    enum waveforms { exp_attenuation };
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

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters,  PerThread* pt)
{
    const numb h = H;
    const int N = 6;
    const numb p[33] = { P(O_N), P(Omega_N), P(zeta), P(K_KC), P(O_beta), P(O_delta), P(kappa_delta), P(K_delta), P(O_3K), P(K_D), P(K_3K), P(Omega_5P), P(F_ex), P(I_Theta), P(omega_I), P(I_bias), P(Omega_C), P(Omega_L), P(O_P), P(K_P), P(C_T), P(rho_A), P(O_2), P(d1), P(d2), P(d3), P(d5), P(rho_C), P(Y_T), P(Omega_C_YS), P(YSfreq), P(YSdel), P(symmetry) };
    numb v[N] = { V(GammaA), V(I), V(C), V(hGate), V(YS), V(t) };

    ifSIGNAL(P(signal), exp_attenuation)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            // time
            Vnext(t) = v[5] + h;

            // YS decay (exact)
            Vnext(YS) = v[4] * exp(-p[29] * h);

            // regular spikes per dt (no per-object state), using p[30] directly
            const int n_spikes =
                (p[30] > 0.0f)
                ? ((int)floor((Vnext(t) - p[31]) * p[30])
                    - (int)floor((v[5] - p[31]) * p[30]))
                : 0;

            // event increment
            Vnext(YS) += (n_spikes > 0)
                ? (n_spikes * (p[27] * p[28] * 1000.0f))
                : 0.0f;

            // dGammaA/dt
            numb dGammaA =
                p[0] * Vnext(YS) * ((numb)1.0 - v[0])
                - p[1] * ((numb)1.0 + (p[2] * v[2] / (v[2] + p[3]))) * v[0];

            // IP3 terms
            numb J_beta = p[4] * v[0];

            numb J_delta =
                p[5] /
                ((numb)1.0 + v[1] / p[6]) *
                ((v[2] * v[2]) / (v[2] * v[2] + p[7] * p[7]));

            numb J_3K =
                p[8] *
                ((v[2] * v[2] * v[2] * v[2]) / (v[2] * v[2] * v[2] * v[2] + p[9] * p[9] * p[9] * p[9])) *
                (v[1] / (v[1] + p[10]));

            numb J_5P = p[11] * v[1];

            numb delta_I = v[1] - p[15];

            numb J_ex =
                -p[12] * (numb)0.5 *
                ((numb)1.0 + tanh((fabs(delta_I) - p[13]) / p[14])) *
                (delta_I > (numb)0.0 ? (numb)1.0 : (delta_I < (numb)0.0 ? (numb)-1.0 : (numb)0.0));

            numb dI = J_beta + J_delta - J_3K - J_5P + J_ex;

            // Ca2+ terms
            numb Q_2 = p[24] * (v[1] + p[23]) / (v[1] + p[25]);
            numb m_inf = (v[1] / (v[1] + p[23])) * (v[2] / (v[2] + p[26]));

            numb h_clipped = fmin(fmax(v[3], (numb)0.0), (numb)1.0);

            numb drive = p[20] - ((numb)1.0 + p[21]) * v[2];

            numb J_r =
                p[16] *
                (m_inf * m_inf * m_inf) * (h_clipped * h_clipped * h_clipped) *
                drive;

            numb J_l = p[17] * drive;

            numb J_p =
                p[18] *
                ((v[2] * v[2]) / (v[2] * v[2] + p[19] * p[19]));

            numb dC = J_r + J_l - J_p;

            // h-gate
            numb h_inf = Q_2 / (Q_2 + v[2]);
            numb tau_h = (numb)1.0 / (p[22] * (Q_2 + v[2]));

            numb dh = (h_inf - h_clipped) / tau_h;

            // Euler updates
            Vnext(GammaA) = v[0] + h * dGammaA;
            Vnext(I) = v[1] + h * dI;
            Vnext(C) = v[2] + h * dC;
            Vnext(hGate) = v[3] + h * dh;
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            // time points 
            numb t0 = v[5];
            numb t_half = v[5] + h * (numb)0.5;
            numb t1 = v[5] + h;

            // spikes on subintervals
            const int n_spikes_01 =
                (p[30] > (numb)0.0)
                ? ((int)floor((t_half - p[31]) * p[30])
                    - (int)floor((t0 - p[31]) * p[30]))
                : 0;

            const int n_spikes_12 =
                (p[30] > (numb)0.0)
                ? ((int)floor((t1 - p[31]) * p[30])
                    - (int)floor((t_half - p[31]) * p[30]))
                : 0;

            // YS values
            numb ys_t0 = v[4];

            numb ys_t_half =
                ys_t0 * exp(-p[29] * h * (numb)0.5)
                + ((n_spikes_01 > 0)
                    ? ((numb)n_spikes_01 * (p[27] * p[28] * (numb)1000.0))
                    : (numb)0.0);

            numb ys_t1 =
                ys_t_half * exp(-p[29] * h * (numb)0.5)
                + ((n_spikes_12 > 0)
                    ? ((numb)n_spikes_12 * (p[27] * p[28] * (numb)1000.0))
                    : (numb)0.0);

            // k1 : RHS at t0
            numb kGamma1 =
                p[0] * ys_t0 * ((numb)1.0 - v[0])
                - p[1] * ((numb)1.0 + p[2] * (v[2] / (v[2] + p[3]))) * v[0];

            numb J_beta1 =
                p[4] * v[0];

            numb J_delta1 =
                p[5] *
                (p[6] / (p[6] + v[1])) *
                ((v[2] * v[2]) / (v[2] * v[2] + p[7] * p[7]));

            numb J_3K1 =
                p[8] *
                ((v[2] * v[2] * v[2] * v[2]) /
                    (v[2] * v[2] * v[2] * v[2] + p[9] * p[9] * p[9] * p[9])) *
                (v[1] / (v[1] + p[10]));

            numb J_5P1 =
                p[11] * v[1];

            numb delta_I1 =
                v[1] - p[15];

            numb J_ex1 =
                -p[12] * (numb)0.5 *
                ((numb)1.0 + tanh((fabs(delta_I1) - p[13]) / p[14])) *
                (delta_I1 > (numb)0.0 ? (numb)1.0 : (delta_I1 < (numb)0.0 ? (numb)-1.0 : (numb)0.0));

            numb kI1 =
                J_beta1 + J_delta1 - J_3K1 - J_5P1 + J_ex1;

            numb Q_21 =
                p[24] * (v[1] + p[23]) / (v[1] + p[25]);

            numb m_inf1 =
                (v[1] / (v[1] + p[23])) * (v[2] / (v[2] + p[26]));

            numb h1 =
                fmin(fmax(v[3], (numb)0.0), (numb)1.0);

            numb drive1 =
                p[20] - ((numb)1.0 + p[21]) * v[2];

            numb J_r1 =
                p[16] *
                (m_inf1 * m_inf1 * m_inf1) *
                (h1 * h1 * h1) *
                drive1;

            numb J_l1 =
                p[17] * drive1;

            numb J_p1 =
                p[18] *
                ((v[2] * v[2]) / (v[2] * v[2] + p[19] * p[19]));

            numb kC1 =
                J_r1 + J_l1 - J_p1;

            numb h_inf1 =
                Q_21 / (Q_21 + v[2]);

            numb tau_h1 =
                (numb)1.0 / (p[22] * (Q_21 + v[2]));

            numb kh1 =
                (h_inf1 - h1) / tau_h1;

            // midpoint estimates
            numb GammaAmp =
                v[0] + (numb)0.5 * h * kGamma1;
            numb Imp =
                v[1] + (numb)0.5 * h * kI1;
            numb Cmp =
                v[2] + (numb)0.5 * h * kC1;
            numb hmp =
                v[3] + (numb)0.5 * h * kh1;

            // k2 : RHS at midpoint
            numb kGamma2 =
                p[0] * ys_t_half * ((numb)1.0 - GammaAmp)
                - p[1] * ((numb)1.0 + p[2] * (Cmp / (Cmp + p[3]))) * GammaAmp;

            numb J_beta2 =
                p[4] * GammaAmp;

            numb J_delta2 =
                p[5] *
                (p[6] / (p[6] + Imp)) *
                ((Cmp * Cmp) / (Cmp * Cmp + p[7] * p[7]));

            numb J_3K2 =
                p[8] *
                ((Cmp * Cmp * Cmp * Cmp) /
                    (Cmp * Cmp * Cmp * Cmp + p[9] * p[9] * p[9] * p[9])) *
                (Imp / (Imp + p[10]));

            numb J_5P2 =
                p[11] * Imp;

            numb delta_I2 =
                Imp - p[15];

            numb J_ex2 =
                -p[12] * (numb)0.5 *
                ((numb)1.0 + tanh((fabs(delta_I2) - p[13]) / p[14])) *
                (delta_I2 > (numb)0.0 ? (numb)1.0 : (delta_I2 < (numb)0.0 ? (numb)-1.0 : (numb)0.0));

            numb kI2 =
                J_beta2 + J_delta2 - J_3K2 - J_5P2 + J_ex2;

            numb Q_22 =
                p[24] * (Imp + p[23]) / (Imp + p[25]);

            numb m_inf2 =
                (Imp / (Imp + p[23])) * (Cmp / (Cmp + p[26]));

            numb h2 =
                fmin(fmax(hmp, (numb)0.0), (numb)1.0);

            numb drive2 =
                p[20] - ((numb)1.0 + p[21]) * Cmp;

            numb J_r2 =
                p[16] *
                (m_inf2 * m_inf2 * m_inf2) *
                (h2 * h2 * h2) *
                drive2;

            numb J_l2 =
                p[17] * drive2;

            numb J_p2 =
                p[18] *
                ((Cmp * Cmp) / (Cmp * Cmp + p[19] * p[19]));

            numb kC2 =
                J_r2 + J_l2 - J_p2;

            numb h_inf2 =
                Q_22 / (Q_22 + Cmp);

            numb tau_h2 =
                (numb)1.0 / (p[22] * (Q_22 + Cmp));

            numb kh2 =
                (h_inf2 - h2) / tau_h2;


            // outputs
            Vnext(t) = t1;
            Vnext(YS) = ys_t1;

            Vnext(GammaA) =
                v[0] + h * kGamma2;

            Vnext(I) =
                v[1] + h * kI2;

            Vnext(C) =
                v[2] + h * kC2;

            Vnext(hGate) =
                v[3] + h * kh2;
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            // time points 
            numb t0 = v[5];
            numb t_half = v[5] + h * (numb)0.5;
            numb t1 = v[5] + h;

            // YS jumps on subintervals 
            const int n_spikes_01 =
                (p[30] > (numb)0.0)
                ? ((int)floor((t_half - p[31]) * p[30])
                    - (int)floor((t0 - p[31]) * p[30]))
                : 0;

            const int n_spikes_12 =
                (p[30] > (numb)0.0)
                ? ((int)floor((t1 - p[31]) * p[30])
                    - (int)floor((t_half - p[31]) * p[30]))
                : 0;

            // YS values at t, t+h/2, t+h 
            numb ys_t0 = v[4];

            numb ys_t_half =
                ys_t0 * exp(-p[29] * h * (numb)0.5)
                + ((n_spikes_01 > 0)
                    ? ((numb)n_spikes_01 * (p[27] * p[28] * (numb)1000.0))
                    : (numb)0.0);

            numb ys_t1 =
                ys_t_half * exp(-p[29] * h * (numb)0.5)
                + ((n_spikes_12 > 0)
                    ? ((numb)n_spikes_12 * (p[27] * p[28] * (numb)1000.0))
                    : (numb)0.0);

            // k1 : RHS at t0
            numb kGamma1 =
                p[0] * ys_t0 * ((numb)1.0 - v[0])
                - p[1] * ((numb)1.0 + p[2] * (v[2] / (v[2] + p[3]))) * v[0];

            numb J_beta1 =
                p[4] * v[0];

            numb J_delta1 =
                p[5] *
                (p[6] / (p[6] + v[1])) *
                ((v[2] * v[2]) / (v[2] * v[2] + p[7] * p[7]));

            numb J_3K1 =
                p[8] *
                ((v[2] * v[2] * v[2] * v[2]) /
                    (v[2] * v[2] * v[2] * v[2] + p[9] * p[9] * p[9] * p[9])) *
                (v[1] / (v[1] + p[10]));

            numb J_5P1 =
                p[11] * v[1];

            numb delta_I1 =
                v[1] - p[15];

            numb J_ex1 =
                -p[12] * (numb)0.5 *
                ((numb)1.0 + tanh((fabs(delta_I1) - p[13]) / p[14])) *
                (delta_I1 > (numb)0.0 ? (numb)1.0 : (delta_I1 < (numb)0.0 ? (numb)-1.0 : (numb)0.0));

            numb kI1 =
                J_beta1 + J_delta1 - J_3K1 - J_5P1 + J_ex1;

            numb Q_21 =
                p[24] * (v[1] + p[23]) / (v[1] + p[25]);

            numb m_inf1 =
                (v[1] / (v[1] + p[23])) * (v[2] / (v[2] + p[26]));

            numb h1 =
                fmin(fmax(v[3], (numb)0.0), (numb)1.0);

            numb drive1 =
                p[20] - ((numb)1.0 + p[21]) * v[2];

            numb J_r1 =
                p[16] *
                (m_inf1 * m_inf1 * m_inf1) *
                (h1 * h1 * h1) *
                drive1;

            numb J_l1 =
                p[17] * drive1;

            numb J_p1 =
                p[18] *
                ((v[2] * v[2]) / (v[2] * v[2] + p[19] * p[19]));

            numb kC1 =
                J_r1 + J_l1 - J_p1;

            numb h_inf1 =
                Q_21 / (Q_21 + v[2]);

            numb tau_h1 =
                (numb)1.0 / (p[22] * (Q_21 + v[2]));

            numb kh1 =
                (h_inf1 - h1) / tau_h1;

            // midpoint state from k1
            numb GammaA_half_1 =
                v[0] + (numb)0.5 * h * kGamma1;
            numb I_half_1 =
                v[1] + (numb)0.5 * h * kI1;
            numb C_half_1 =
                v[2] + (numb)0.5 * h * kC1;
            numb hGate_half_1 =
                v[3] + (numb)0.5 * h * kh1;

            // k2 : RHS at t0 + h/2 using ys_t_half
            numb kGamma2 =
                p[0] * ys_t_half * ((numb)1.0 - GammaA_half_1)
                - p[1] * ((numb)1.0 + p[2] * (C_half_1 / (C_half_1 + p[3]))) * GammaA_half_1;

            numb J_beta2 =
                p[4] * GammaA_half_1;

            numb J_delta2 =
                p[5] *
                (p[6] / (p[6] + I_half_1)) *
                ((C_half_1 * C_half_1) / (C_half_1 * C_half_1 + p[7] * p[7]));

            numb J_3K2 =
                p[8] *
                ((C_half_1 * C_half_1 * C_half_1 * C_half_1) /
                    (C_half_1 * C_half_1 * C_half_1 * C_half_1 + p[9] * p[9] * p[9] * p[9])) *
                (I_half_1 / (I_half_1 + p[10]));

            numb J_5P2 =
                p[11] * I_half_1;

            numb delta_I2 =
                I_half_1 - p[15];

            numb J_ex2 =
                -p[12] * (numb)0.5 *
                ((numb)1.0 + tanh((fabs(delta_I2) - p[13]) / p[14])) *
                (delta_I2 > (numb)0.0 ? (numb)1.0 : (delta_I2 < (numb)0.0 ? (numb)-1.0 : (numb)0.0));

            numb kI2 =
                J_beta2 + J_delta2 - J_3K2 - J_5P2 + J_ex2;

            numb Q_22 =
                p[24] * (I_half_1 + p[23]) / (I_half_1 + p[25]);

            numb m_inf2 =
                (I_half_1 / (I_half_1 + p[23])) * (C_half_1 / (C_half_1 + p[26]));

            numb h2 =
                fmin(fmax(hGate_half_1, (numb)0.0), (numb)1.0);

            numb drive2 =
                p[20] - ((numb)1.0 + p[21]) * C_half_1;

            numb J_r2 =
                p[16] *
                (m_inf2 * m_inf2 * m_inf2) *
                (h2 * h2 * h2) *
                drive2;

            numb J_l2 =
                p[17] * drive2;

            numb J_p2 =
                p[18] *
                ((C_half_1 * C_half_1) / (C_half_1 * C_half_1 + p[19] * p[19]));

            numb kC2 =
                J_r2 + J_l2 - J_p2;

            numb h_inf2 =
                Q_22 / (Q_22 + C_half_1);

            numb tau_h2 =
                (numb)1.0 / (p[22] * (Q_22 + C_half_1));

            numb kh2 =
                (h_inf2 - h2) / tau_h2;

            // midpoint state from k2
            numb GammaA_half_2 =
                v[0] + (numb)0.5 * h * kGamma2;
            numb I_half_2 =
                v[1] + (numb)0.5 * h * kI2;
            numb C_half_2 =
                v[2] + (numb)0.5 * h * kC2;
            numb hGate_half_2 =
                v[3] + (numb)0.5 * h * kh2;

            // k3 : RHS at t0 + h/2 using ys_t_half
            numb kGamma3 =
                p[0] * ys_t_half * ((numb)1.0 - GammaA_half_2)
                - p[1] * ((numb)1.0 + p[2] * (C_half_2 / (C_half_2 + p[3]))) * GammaA_half_2;

            numb J_beta3 =
                p[4] * GammaA_half_2;

            numb J_delta3 =
                p[5] *
                (p[6] / (p[6] + I_half_2)) *
                ((C_half_2 * C_half_2) / (C_half_2 * C_half_2 + p[7] * p[7]));

            numb J_3K3 =
                p[8] *
                ((C_half_2 * C_half_2 * C_half_2 * C_half_2) /
                    (C_half_2 * C_half_2 * C_half_2 * C_half_2 + p[9] * p[9] * p[9] * p[9])) *
                (I_half_2 / (I_half_2 + p[10]));

            numb J_5P3 =
                p[11] * I_half_2;

            numb delta_I3 =
                I_half_2 - p[15];

            numb J_ex3 =
                -p[12] * (numb)0.5 *
                ((numb)1.0 + tanh((fabs(delta_I3) - p[13]) / p[14])) *
                (delta_I3 > (numb)0.0 ? (numb)1.0 : (delta_I3 < (numb)0.0 ? (numb)-1.0 : (numb)0.0));

            numb kI3 =
                J_beta3 + J_delta3 - J_3K3 - J_5P3 + J_ex3;

            numb Q_23 =
                p[24] * (I_half_2 + p[23]) / (I_half_2 + p[25]);

            numb m_inf3 =
                (I_half_2 / (I_half_2 + p[23])) * (C_half_2 / (C_half_2 + p[26]));

            numb h3 =
                fmin(fmax(hGate_half_2, (numb)0.0), (numb)1.0);

            numb drive3 =
                p[20] - ((numb)1.0 + p[21]) * C_half_2;

            numb J_r3 =
                p[16] *
                (m_inf3 * m_inf3 * m_inf3) *
                (h3 * h3 * h3) *
                drive3;

            numb J_l3 =
                p[17] * drive3;

            numb J_p3 =
                p[18] *
                ((C_half_2 * C_half_2) / (C_half_2 * C_half_2 + p[19] * p[19]));

            numb kC3 =
                J_r3 + J_l3 - J_p3;

            numb h_inf3 =
                Q_23 / (Q_23 + C_half_2);

            numb tau_h3 =
                (numb)1.0 / (p[22] * (Q_23 + C_half_2));

            numb kh3 =
                (h_inf3 - h3) / tau_h3;

            // endpoint state from k3
            numb GammaA_end =
                v[0] + h * kGamma3;
            numb I_end =
                v[1] + h * kI3;
            numb C_end =
                v[2] + h * kC3;
            numb hGate_end =
                v[3] + h * kh3;

            // k4 : RHS at t0 + h using ys_t1
            numb kGamma4 =
                p[0] * ys_t1 * ((numb)1.0 - GammaA_end)
                - p[1] * ((numb)1.0 + p[2] * (C_end / (C_end + p[3]))) * GammaA_end;

            numb J_beta4 =
                p[4] * GammaA_end;

            numb J_delta4 =
                p[5] *
                (p[6] / (p[6] + I_end)) *
                ((C_end * C_end) / (C_end * C_end + p[7] * p[7]));

            numb J_3K4 =
                p[8] *
                ((C_end * C_end * C_end * C_end) /
                    (C_end * C_end * C_end * C_end + p[9] * p[9] * p[9] * p[9])) *
                (I_end / (I_end + p[10]));

            numb J_5P4 =
                p[11] * I_end;

            numb delta_I4 =
                I_end - p[15];

            numb J_ex4 =
                -p[12] * (numb)0.5 *
                ((numb)1.0 + tanh((fabs(delta_I4) - p[13]) / p[14])) *
                (delta_I4 > (numb)0.0 ? (numb)1.0 : (delta_I4 < (numb)0.0 ? (numb)-1.0 : (numb)0.0));

            numb kI4 =
                J_beta4 + J_delta4 - J_3K4 - J_5P4 + J_ex4;

            numb Q_24 =
                p[24] * (I_end + p[23]) / (I_end + p[25]);

            numb m_inf4 =
                (I_end / (I_end + p[23])) * (C_end / (C_end + p[26]));

            numb h4 =
                fmin(fmax(hGate_end, (numb)0.0), (numb)1.0);

            numb drive4 =
                p[20] - ((numb)1.0 + p[21]) * C_end;

            numb J_r4 =
                p[16] *
                (m_inf4 * m_inf4 * m_inf4) *
                (h4 * h4 * h4) *
                drive4;

            numb J_l4 =
                p[17] * drive4;

            numb J_p4 =
                p[18] *
                ((C_end * C_end) / (C_end * C_end + p[19] * p[19]));

            numb kC4 =
                J_r4 + J_l4 - J_p4;

            numb h_inf4 =
                Q_24 / (Q_24 + C_end);

            numb tau_h4 =
                (numb)1.0 / (p[22] * (Q_24 + C_end));

            numb kh4 =
                (h_inf4 - h4) / tau_h4;

            // outputs
            Vnext(t) = t1;
            Vnext(YS) = ys_t1;

            Vnext(GammaA) =
                v[0] + h * (kGamma1 + (numb)2.0 * kGamma2 + (numb)2.0 * kGamma3 + kGamma4) / (numb)6.0;

            Vnext(I) =
                v[1] + h * (kI1 + (numb)2.0 * kI2 + (numb)2.0 * kI3 + kI4) / (numb)6.0;

            Vnext(C) =
                v[2] + h * (kC1 + (numb)2.0 * kC2 + (numb)2.0 * kC3 + kC4) / (numb)6.0;

            Vnext(hGate) =
                v[3] + h * (kh1 + (numb)2.0 * kh2 + (numb)2.0 * kh3 + kh4) / (numb)6.0;
        }

       /* ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = (numb)0.5 * h - p[32];
            numb h2 = (numb)0.5 * h + p[32];

            // time points
            numb t0 = v[5];
            numb t_mid = v[5] + h1;
            numb t1 = v[5] + h;

            // spikes on subintervals
            const int n_spikes_01 =
                (p[30] > (numb)0.0)
                ? ((int)floor((t_mid - p[31]) * p[30])
                    - (int)floor((t0 - p[31]) * p[30]))
                : 0;

            const int n_spikes_12 =
                (p[30] > (numb)0.0)
                ? ((int)floor((t1 - p[31]) * p[30])
                    - (int)floor((t_mid - p[31]) * p[30]))
                : 0;

            // YS values 
            numb ys_t0 = v[4];

            numb ys_t_mid =
                ys_t0 * exp(-p[29] * h1)
                + ((n_spikes_01 > 0)
                    ? ((numb)n_spikes_01 * (p[27] * p[28] * (numb)1000.0))
                    : (numb)0.0);

            numb ys_t1 =
                ys_t_mid * exp(-p[29] * h2)
                + ((n_spikes_12 > 0)
                    ? ((numb)n_spikes_12 * (p[27] * p[28] * (numb)1000.0))
                    : (numb)0.0);

            // RHS at current state (first substep)
            numb dGammaA =
                p[0] * ys_t0 * ((numb)1.0 - v[0])
                - p[1] * ((numb)1.0 + p[2] * (v[2] / (v[2] + p[3]))) * v[0];

            numb J_beta =
                p[4] * v[0];

            numb J_delta =
                p[5] *
                (p[6] / (p[6] + v[1])) *
                ((v[2] * v[2]) / (v[2] * v[2] + p[7] * p[7]));

            numb J_3K =
                p[8] *
                ((v[2] * v[2] * v[2] * v[2]) /
                    (v[2] * v[2] * v[2] * v[2] + p[9] * p[9] * p[9] * p[9])) *
                (v[1] / (v[1] + p[10]));

            numb J_5P =
                p[11] * v[1];

            numb delta_I =
                v[1] - p[15];

            numb J_ex =
                -p[12] * (numb)0.5 *
                ((numb)1.0 + tanh((fabs(delta_I) - p[13]) / p[14])) *
                (delta_I > (numb)0.0 ? (numb)1.0 : (delta_I < (numb)0.0 ? (numb)-1.0 : (numb)0.0));

            numb dI =
                J_beta + J_delta - J_3K - J_5P + J_ex;

            numb Q_2 =
                p[24] * (v[1] + p[23]) / (v[1] + p[25]);

            numb m_inf =
                (v[1] / (v[1] + p[23])) * (v[2] / (v[2] + p[26]));

            numb h_clipped =
                fmin(fmax(v[3], (numb)0.0), (numb)1.0);

            numb drive =
                p[20] - ((numb)1.0 + p[21]) * v[2];

            numb J_r =
                p[16] *
                (m_inf * m_inf * m_inf) *
                (h_clipped * h_clipped * h_clipped) *
                drive;

            numb J_l =
                p[17] * drive;

            numb J_p =
                p[18] *
                ((v[2] * v[2]) / (v[2] * v[2] + p[19] * p[19]));

            numb dC =
                J_r + J_l - J_p;

            numb h_inf =
                Q_2 / (Q_2 + v[2]);

            numb tau_h =
                (numb)1.0 / (p[22] * (Q_2 + v[2]));

            numb dh =
                (h_inf - h_clipped) / tau_h;

            // first asymmetric substep
            numb GammaAmp =
                v[0] + h1 * dGammaA;
            numb Imp =
                v[1] + h1 * dI;
            numb Cmp =
                v[2] + h1 * dC;
            numb hmp =
                v[3] + h1 * dh;

            // RHS at intermediate state (second substep)
            dGammaA =
                p[0] * ys_t_mid * ((numb)1.0 - GammaAmp)
                - p[1] * ((numb)1.0 + p[2] * (Cmp / (Cmp + p[3]))) * GammaAmp;

            J_beta =
                p[4] * GammaAmp;

            J_delta =
                p[5] *
                (p[6] / (p[6] + Imp)) *
                ((Cmp * Cmp) / (Cmp * Cmp + p[7] * p[7]));

            J_3K =
                p[8] *
                ((Cmp * Cmp * Cmp * Cmp) /
                    (Cmp * Cmp * Cmp * Cmp + p[9] * p[9] * p[9] * p[9])) *
                (Imp / (Imp + p[10]));

            J_5P =
                p[11] * Imp;

            delta_I =
                Imp - p[15];

            J_ex =
                -p[12] * (numb)0.5 *
                ((numb)1.0 + tanh((fabs(delta_I) - p[13]) / p[14])) *
                (delta_I > (numb)0.0 ? (numb)1.0 : (delta_I < (numb)0.0 ? (numb)-1.0 : (numb)0.0));

            dI =
                J_beta + J_delta - J_3K - J_5P + J_ex;

            Q_2 =
                p[24] * (Imp + p[23]) / (Imp + p[25]);

            m_inf =
                (Imp / (Imp + p[23])) * (Cmp / (Cmp + p[26]));

            h_clipped =
                fmin(fmax(hmp, (numb)0.0), (numb)1.0);

            drive =
                p[20] - ((numb)1.0 + p[21]) * Cmp;

            J_r =
                p[16] *
                (m_inf * m_inf * m_inf) *
                (h_clipped * h_clipped * h_clipped) *
                drive;

            J_l =
                p[17] * drive;

            J_p =
                p[18] *
                ((Cmp * Cmp) / (Cmp * Cmp + p[19] * p[19]));

            dC =
                J_r + J_l - J_p;

            h_inf =
                Q_2 / (Q_2 + Cmp);

            tau_h =
                (numb)1.0 / (p[22] * (Q_2 + Cmp));

            dh =
                (h_inf - h_clipped) / tau_h;

            // second asymmetric substep to final state
            Vnext(t) = t1;
            Vnext(YS) = ys_t1;

            Vnext(GammaA) =
                GammaAmp + h2 * dGammaA;
            Vnext(I) =
                Imp + h2 * dI;
            Vnext(C) =
                Cmp + h2 * dC;
            Vnext(hGate) =
                hmp + h2 * dh;
        }*/
    }
}

#undef name