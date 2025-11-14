#include "main.h"
#include "wilson.h"

#define name wilson

namespace attributes
{
    enum variables { v, r, i, t };
    enum parameters { C, tau, p0, p1, p2, p3, p4, p5, p6, p7, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, method, COUNT };
    enum waveforms { square, sine, triangle };
    enum methods { ExplicitEuler, SemiExplicitEuler, ImplicitEuler, ExplicitMidpoint, ImplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
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
    ifSIGNAL(P(signal), square)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
            Vnext(t) = V(t) + H;
            Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
            Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
        }
		ifMETHOD(P(method), SemiExplicitEuler)
        {
            Vnext(t) = V(t) + H;
			Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * Vnext(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
        }
		ifMETHOD(P(method),ImplicitEuler)
        {
			Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			Vnext(t)= V(t) + H;
			numb v_guess = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			numb r_guess = V(r) + H * ((1.0f / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
	
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));	
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));

			Vnext(v) = v_guess;
			Vnext(r) = r_guess;
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
		ifMETHOD(P(method),ImplicitMidpoint)
        {
			numb imp = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			numb tmp= V(t) + 0.5f * H;
			numb v_mid_guess = V(v) + H * 0.5 *((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
			numb r_mid_guess = V(r) + H * 0.5 * ((1.0f / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			v_mid_guess = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			
			r_mid_guess = V(r) + H * 0.5 *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5 *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5 *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5 *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5 *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			
			Vnext(t) = V(t) + H;
			Vnext(i) = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1.0f / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * ((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
        }
		ifMETHOD(P(method), ExplicitRungeKutta4)
        {
			numb i1 = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			numb kt1 = V(t) + 0.5f * H;
			
			numb kv1 = (-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + i1) / P(C);
			numb kr1 = (1.0f / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7));
			
			numb vmp = V(v) + 0.5f * H * kv1;
			numb rmp = V(r) + 0.5f * H * kr1;
			
			numb i2 = P(Idc) + (fmodf((kt1 - P(Idel)) > 0 ? (kt1 - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - kt1), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			numb kv2 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr2 = (1.0f / P(tau)) * (-rmp + P(p6) * vmp + P(p7));
			
			vmp = V(v) + 0.5f * H * kv2;
			rmp = V(r) + 0.5f * H * kr2;
			
			numb kv3 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr3 = (1.0f / P(tau)) * (-rmp + P(p6) * vmp + P(p7));
			Vnext(t) = V(t) + H;
			
			vmp = V(v) + H * kv3;
			rmp = V(r) + H * kr3;
			
			numb i3 = P(Idc) + (fmodf((Vnext(t) - P(Idel)) > 0 ? (Vnext(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - Vnext(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			numb kv4 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i3) / P(C);
			numb kr4 = (1.0f / P(tau)) * (-rmp + P(p6) * vmp + P(p7));

			Vnext(i) = i3;
			Vnext(v) = V(v) + H * (kv1 + 2.0f * kv2 + 2.0f * kv3 + kv4) / 6.0f;
			Vnext(r) = V(r) + H * (kr1 + 2.0f * kr2 + 2.0f * kr3 + kr4) / 6.0f;	
        }
		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = 0.5 * H - P(symmetry);
			numb h2 = 0.5 * H + P(symmetry);
			
			Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			Vnext(v) = V(v) + h1 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + h1 * ((1.0 / P(tau)) * (-V(r) + P(p6) * Vnext(v) + P(p7)));
			
			
			numb vmp = Vnext(v);
			numb rmp = Vnext(r);
			
			Vnext(r) =  (rmp +(h2/P(tau))*(P(p6)*vmp + P(p7)))/(1 + h2/P(tau));

			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));		
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(t) = V(t) + H;
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
		ifMETHOD(P(method), SemiExplicitEuler)
        {
            Vnext(t) = V(t) + H;
			Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
			Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (Vnext(t) - P(Idel)));
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * Vnext(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
        }
		ifMETHOD(P(method),ImplicitEuler)
        {
			Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
			Vnext(t)= V(t) + H;
			numb v_guess = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			numb r_guess = V(r) + H * ((1.0f / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
	
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));	
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));

			Vnext(v) = v_guess;
			Vnext(r) = r_guess;
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
		ifMETHOD(P(method),ImplicitMidpoint)
        {
			numb imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
			numb tmp= V(t) + 0.5f * H;
			numb v_mid_guess = V(v) + H * 0.5 *((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
			numb r_mid_guess = V(r) + H * 0.5 * ((1.0f / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			v_mid_guess = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			
			r_mid_guess = V(r) + H * 0.5 *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5 *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5 *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5 *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5 *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			
			Vnext(t) = V(t) + H;
			Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * ((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
        }
		ifMETHOD(P(method), ExplicitRungeKutta4)
        {
			numb i1 = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
			numb kt1 = V(t) + 0.5f * H;
			
			numb kv1 = (-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + i1) / P(C);
			numb kr1 = (1.0f / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7));
			
			numb vmp = V(v) + 0.5f * H * kv1;
			numb rmp = V(r) + 0.5f * H * kr1;
			
			numb i2 = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (kt1 - P(Idel)));
			numb kv2 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr2 = (1.0f / P(tau)) * (-rmp + P(p6) * vmp + P(p7));
			
			vmp = V(v) + 0.5f * H * kv2;
			rmp = V(r) + 0.5f * H * kr2;
			
			numb kv3 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr3 = (1.0f / P(tau)) * (-rmp + P(p6) * vmp + P(p7));
			Vnext(t) = V(t) + H;
			
			vmp = V(v) + H * kv3;
			rmp = V(r) + H * kr3;
			
			numb i3 = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (Vnext(t) - P(Idel)));
			numb kv4 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i3) / P(C);
			numb kr4 = (1.0f / P(tau)) * (-rmp + P(p6) * vmp + P(p7));

			Vnext(i) = i3;
			Vnext(v) = V(v) + H * (kv1 + 2.0f * kv2 + 2.0f * kv3 + kv4) / 6.0f;
			Vnext(r) = V(r) + H * (kr1 + 2.0f * kr2 + 2.0f * kr3 + kr4) / 6.0f;
		}
		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = 0.5 * H - P(symmetry);
			numb h2 = 0.5 * H + P(symmetry);
			
			Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
			Vnext(v) = V(v) + h1 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + h1 * ((1.0 / P(tau)) * (-V(r) + P(p6) * Vnext(v) + P(p7)));
			
			
			numb vmp = Vnext(v);
			numb rmp = Vnext(r);
			
			Vnext(r) =  (rmp +(h2/P(tau))*(P(p6)*vmp + P(p7)))/(1 + h2/P(tau));

			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));		
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(t) = V(t) + H;
		}
    }
    ifSIGNAL(P(signal), triangle)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
            Vnext(t) = V(t) + H;
            Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
            Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
        }
		ifMETHOD(P(method), SemiExplicitEuler)
        {
            Vnext(t) = V(t) + H;
			Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
			Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (Vnext(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (Vnext(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (Vnext(t) - P(Idel)) + 1.0f) / 2.0f)));
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * Vnext(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
        }
		ifMETHOD(P(method), ImplicitEuler)
        {
			Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
			Vnext(t)= V(t) + H;
			numb v_guess = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			numb r_guess = V(r) + H * ((1.0f / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
	
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));	
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * ((1.0f / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));

			Vnext(v) = v_guess;
			Vnext(r) = r_guess;
        }
        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
            numb tmp = V(t) + H * 0.5f;
            numb vmp = V(v) + H * 0.5f * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
            numb rmp = V(r) + H * 0.5f * ((1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

            Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));
            Vnext(t) = V(t) + H;
            Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + Vnext(i)) / P(C));
            Vnext(r) = V(r) + H * ((1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7)));
        }
		ifMETHOD(P(method),ImplicitMidpoint)
        {
			numb imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
			numb tmp= V(t) + 0.5f * H;
			numb v_mid_guess = V(v) + H * 0.5f *((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
			numb r_mid_guess = V(r) + H * 0.5f * ((1.0f / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			v_mid_guess = V(v) + H * 0.5f * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5f * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5f * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5f * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * 0.5f * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			
			r_mid_guess = V(r) + H * 0.5f *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5f *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5f *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5f *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * 0.5f *((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			
			Vnext(t) = V(t) + H;
			Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4 * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * ((1.0f / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
        }
		ifMETHOD(P(method), ExplicitRungeKutta4) 
		{
			numb i1 = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
			numb kt1 = V(t) + 0.5f * H;
			
			numb kv1 = (-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + i1) / P(C);
			numb kr1 = (1.0f / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7));
			
			numb vmp = V(v) + 0.5f * H * kv1;
			numb rmp = V(r) + 0.5f * H * kr1;
			
			numb i2 = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (kt1 - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (kt1 - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (kt1 - P(Idel)) + 1.0f) / 2.0f)));
			numb kv2 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr2 = (1.0f / P(tau)) * (-rmp + P(p6) * vmp + P(p7));
			
			vmp = V(v) + 0.5f * H * kv2;
			rmp = V(r) + 0.5f * H * kr2;
			
			numb kv3 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr3 = (1.0f / P(tau)) * (-rmp + P(p6) * vmp + P(p7));
			Vnext(t) = V(t) + H;
			
			vmp = V(v) + H * kv3;
			rmp = V(r) + H * kr3;
			
			numb i3 = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (Vnext(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (Vnext(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (Vnext(t) - P(Idel)) + 1.0f) / 2.0f)));
			numb kv4 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i3) / P(C);
			numb kr4 = (1.0f / P(tau)) * (-rmp + P(p6) * vmp + P(p7));

			Vnext(t) = V(t) + H;
			Vnext(i) = i3;
			Vnext(v) = V(v) + H * (kv1 + 2.0f * kv2 + 2.0f * kv3 + kv4) / 6.0f;
			Vnext(r) = V(r) + H * (kr1 + 2.0f * kr2 + 2.0f * kr3 + kr4) / 6.0f;
	
		}
		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = 0.5 * H - P(symmetry);
			numb h2 = 0.5 * H + P(symmetry);
			
			Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
			Vnext(v) = V(v) + h1 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + h1 * ((1.0 / P(tau)) * (-V(r) + P(p6) * Vnext(v) + P(p7)));
			
			numb vmp = Vnext(v);
			numb rmp = Vnext(r);
			
			Vnext(r) =  (rmp +(h2/P(tau))*(P(p6)*vmp + P(p7)))/(1 + h2/P(tau));

			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));		
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(v) = vmp + h2 * ((-(P(p0) + P(p1) * Vnext(v) + P(p2) * Vnext(v) * Vnext(v)) * (Vnext(v) - P(p3)) - P(p5) * Vnext(r) * (Vnext(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(t) = V(t) + H;
		}
    }
}

#undef name