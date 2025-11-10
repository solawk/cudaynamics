#include "main.h"
#include "izhikevich.h"

namespace attributes
{
    enum variables { v, u, i, t };
    enum parameters { a, b, c, d, p0, p1, p2, p3, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, method, COUNT };
	enum waveforms { square };
    enum methods { ExplicitEuler, SemiExplicitEuler, ImplicitEuler, ExplicitMidpoint, ImplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_izhikevich(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_izhikevich);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_izhikevich(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, variation, &finiteDifferenceScheme_izhikevich, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MINMAX_Settings max_settings(MS(MAX, 0));
        MAX(data, variation, &finiteDifferenceScheme_izhikevich, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBSCAN_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4),
            H);
        Period(data, variation, &finiteDifferenceScheme_izhikevich, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__  void finiteDifferenceScheme_izhikevich(numb* currentV, numb* nextV, numb* parameters)
{	ifSIGNAL(P(signal), square)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			Vnext(t) = V(t) + H;
			Vnext(v) = V(v) + H * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(i));
			Vnext(u) = V(u) + H * (P(a) * (P(b) * V(v) - V(u)));

			if (Vnext(v) >= P(p3))
			{
				Vnext(v) = P(c);
				Vnext(u) = Vnext(u) + P(d);
			}
		}
		ifMETHOD(P(method), SemiExplicitEuler)
		{
			Vnext(t) = V(t) + H;
			Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			Vnext(v) = V(v) + H * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(i));
			Vnext(u) = V(u) + H * (P(a) * (P(b) * Vnext(v) - V(u)));
			if (Vnext(v) >= P(p3))
			{
				Vnext(v) = P(c);
				Vnext(u) = Vnext(u) + P(d);
			}
		}
        ifMETHOD(P(method), ImplicitEuler)
        {
            Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(t) = V(t) + H;

            numb v_guess = V(v) + H * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(i));
            numb u_guess = V(u) + H * (P(a) * (P(b) * Vnext(v) - V(u)));

            v_guess = V(v) + H * (P(p0) * v_guess * v_guess + P(p1) * v_guess + P(p2) - u_guess + Vnext(i));
            v_guess = V(v) + H * (P(p0) * v_guess * v_guess + P(p1) * v_guess + P(p2) - u_guess + Vnext(i));
            v_guess = V(v) + H * (P(p0) * v_guess * v_guess + P(p1) * v_guess + P(p2) - u_guess + Vnext(i));
            v_guess = V(v) + H * (P(p0) * v_guess * v_guess + P(p1) * v_guess + P(p2) - u_guess + Vnext(i));

            u_guess = V(u) + H * (P(a) * (P(b) * v_guess - u_guess));
            u_guess = V(u) + H * (P(a) * (P(b) * v_guess - u_guess));
            u_guess = V(u) + H * (P(a) * (P(b) * v_guess - u_guess));
            u_guess = V(u) + H * (P(a) * (P(b) * v_guess - u_guess));

            if (v_guess >= P(p3))
            {
                v_guess = P(c);
                u_guess = u_guess + P(d);
            }

            Vnext(v) = v_guess;
            Vnext(u) = u_guess;
        }
        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            numb tmp = V(t) + H * 0.5;
            numb vmp = V(v) + H * 0.5 * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(i));
            numb ump = V(u) + H * 0.5 * (P(a) * (P(b) * Vnext(v) - V(u)));

            Vnext(i) = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(t) = V(t) + H;
            Vnext(v) = V(v) + H * (P(p0) * vmp * vmp + P(p1) * vmp + P(p2) - ump + Vnext(i));
            Vnext(u) = V(u) + H * (P(a) * (P(b) * vmp - ump));

            if (Vnext(v)  >= P(p3))
            {
                Vnext(v) = P(c);
                Vnext(u) = Vnext(u) + P(d);
            }
        }
        ifMETHOD(P(method), ImplicitMidpoint)
        {
            numb imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
            numb tmp = V(t) + 0.5f * H;
            numb v_mid_guess = V(v) + H * 0.5f * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + imp);
            numb u_mid_guess = V(u) + H * 0.5f * (P(a) * (P(b) * Vnext(v) - V(u)));

            v_mid_guess = V(v) + H * 0.5f * (P(p0) * v_mid_guess * v_mid_guess + P(p1) * v_mid_guess + P(p2) - u_mid_guess + imp);
            v_mid_guess = V(v) + H * 0.5f * (P(p0) * v_mid_guess * v_mid_guess + P(p1) * v_mid_guess + P(p2) - u_mid_guess + imp);
            v_mid_guess = V(v) + H * 0.5f * (P(p0) * v_mid_guess * v_mid_guess + P(p1) * v_mid_guess + P(p2) - u_mid_guess + imp);
            v_mid_guess = V(v) + H * 0.5f * (P(p0) * v_mid_guess * v_mid_guess + P(p1) * v_mid_guess + P(p2) - u_mid_guess + imp);

            u_mid_guess = V(u) + H * 0.5f * (P(a) * (P(b) * v_mid_guess - u_mid_guess));
            u_mid_guess = V(u) + H * 0.5f * (P(a) * (P(b) * v_mid_guess - u_mid_guess));
            u_mid_guess = V(u) + H * 0.5f * (P(a) * (P(b) * v_mid_guess - u_mid_guess));
            u_mid_guess = V(u) + H * 0.5f * (P(a) * (P(b) * v_mid_guess - u_mid_guess));

            Vnext(t) = V(t) + H;
            Vnext(i) = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1.0f / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(v) = V(v) + H * (P(p0) * v_mid_guess * v_mid_guess + P(p1) * v_mid_guess + P(p2) - u_mid_guess + Vnext(i));
            Vnext(u) = V(u) + H * (P(a) * (P(b) * v_mid_guess - u_mid_guess));

            if (Vnext(v) >= P(p3))
            {
                Vnext(v) = P(c);
                Vnext(u) = Vnext(u) + P(d);
            }
        }
        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb i1 = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            numb kt1 = V(t) + 0.5f * H;

            numb kv1 = (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + i1);
            numb ku1 = (P(a) * (P(b) * V(v) - V(u)));

            numb vmp = V(v) + 0.5f * H * kv1;
            numb ump = V(u) + 0.5f * H * ku1;

            numb i2 = P(Idc) + (fmodf((kt1 - P(Idel)) > 0 ? (kt1 - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - kt1), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            numb kv2 = (P(p0) * vmp * vmp + P(p1) * vmp + P(p2) - ump + i2);
            numb ku2 = (P(a) * (P(b) * vmp - ump));

            vmp = V(v) + 0.5f * H * kv2;
            ump = V(u) + 0.5f * H * ku2;
             
            numb kv3 = (P(p0) * vmp * vmp + P(p1) * vmp + P(p2) - ump + i2);
            numb ku3 = (P(a) * (P(b) * vmp - ump));
            Vnext(t) = V(t) + H;

            vmp = V(v) + 0.5f * H * kv2;
            ump = V(u) + 0.5f * H * ku2;

            numb i3 = P(Idc) + (fmodf((Vnext(t) - P(Idel)) > 0 ? (Vnext(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - Vnext(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            numb kv4 = (P(p0) * vmp * vmp + P(p1) * vmp + P(p2) - ump + i3);
            numb ku4 = (P(a) * (P(b) * vmp - ump));

            Vnext(i) = i3;
            Vnext(v) = V(v) + H * (kv1 + 2.0f * kv2 + 2.0f * kv3 + kv4) / 6.0f;
            Vnext(u) = V(u) + H * (ku1 + 2.0f * ku2 + 2.0f * ku3 + ku4) / 6.0f;

            if (Vnext(v) >= P(p3))
            {
                Vnext(v) = P(c);
                Vnext(u) = Vnext(u) + P(d);
            }
        }
        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = 0.5 * H - P(symmetry);
            numb h2 = 0.5 * H + P(symmetry);

            Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(v) = V(v) + h1 * (P(p0) * V(v) * V(v) + P(p1) * V(v) + P(p2) - V(u) + Vnext(i));
            Vnext(u) = V(u) + h1 * (P(a) * (P(b) * Vnext(v) - V(u)));

            numb vmp = Vnext(v);
            numb ump = Vnext(u);

            Vnext(u) = (ump + h2 * P(b) * P(a) * vmp) / (1.0 + h2* P(a));

            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(v) = vmp + h2 * (P(p0) * Vnext(v) * Vnext(v) + P(p1) * Vnext(v) + P(p2) - Vnext(u) + Vnext(i));
            Vnext(t) = V(t) + H;


            if (Vnext(v) >= P(p3))
            {
                Vnext(v) = P(c);
                Vnext(u) = Vnext(u) + P(d);
            }
        }
	}
}