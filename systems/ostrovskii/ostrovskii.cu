#include "main.h"
#include "ostrovskii.h"

#define name ostrovskii

namespace attributes
{
    enum variables { v, x, i, t};
    enum parameters { 
		// RHS
		Utd, Uvm, C,
		// GI403
		Is, Vt, Vp, Ip, Iv, D, E,
		// AND_TS
		Ron_p, Ron_n, Vth_p, Vh_p, Vth_n, Vh_n, tau_s, tau_r, Vs, Vr, A, Ds, Dr, Ilk, 
		//Stuff
		Idc, Iamp, Ifreq, Idel, Idf, 
		signal, method, 
		COUNT 
	};
    enum waveforms { square, sine, triangle };
	enum methods { ExplicitRungeKutta4 };
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

__device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters)
{
	ifSIGNAL(P(signal), square) {
		ifMETHOD(P(method), ExplicitRungeKutta4) {
			numb Im, Id;

			Id = P(Is) * (exp((V(v) + P(Utd)) / P(Vt)) - exp(-(V(v) + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (V(v) + P(Utd)) * exp(-(V(v) + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (V(v) + P(Utd) - P(E))) + atan(P(D) * (V(v) + P(Utd) + P(E))));

			if ((-V(v) + P(Uvm)) > 0)
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_p) + P(Ilk);
			else
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_n) - P(Ilk);

			numb imp = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv1 = (imp + Im - Id) / P(C);
			numb kx1 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-V(v) + P(Uvm)) - P(Vth_p)) * ((-V(v) + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * V(x) + P(Ds))))) * (1 - V(x)) + V(x) * (1 - 1 / (exp(P(A) * (1 - V(x)))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-V(v) + P(Uvm)) - P(Vh_n)) * ((-V(v) + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * V(x))))) * (1 - V(x)) + V(x) * (1 - 1 / (exp(P(A) * (1 - V(x)) + P(Dr)))));

			numb vmp = V(v) + 0.5f * H * kv1;
			numb xmp = V(x) + 0.5f * H * kx1;
			numb tmp = V(t) + 0.5f * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > 0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv2 = (imp + Im - Id) / P(C);
			numb kx2 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * xmp + P(Ds))))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * xmp)))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp) + P(Dr)))));

			vmp = V(v) + 0.5f * H * kv2;
			xmp = V(x) + 0.5f * H * kx2;
			tmp = V(t) + 0.5f * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > 0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv3 = (imp + Im - Id) / P(C);
			numb kx3 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * xmp + P(Ds))))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * xmp)))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp) + P(Dr)))));

			vmp = V(v) + H * kv3;
			xmp = V(x) + H * kx3;
			tmp = V(t) + H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > 0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			Vnext(i) = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv4 = (Vnext(i) + Im - Id) / P(C);
			numb kx4 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * xmp + P(Ds))))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * xmp)))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp) + P(Dr)))));

			Vnext(v) = V(v) + H * (kv1 + 2.0f * kv2 + 2.0f * kv3 + kv4) / 6.0f;
			Vnext(x) = V(x) + H * (kx1 + 2.0f * kx2 + 2.0f * kx3 + kx4) / 6.0f;
			Vnext(t) = V(t) + H;
		}
	}
	ifSIGNAL(P(signal), sine) {
		ifMETHOD(P(method), ExplicitRungeKutta4) {
			numb Im, Id;

			Id = P(Is) * (exp((V(v) + P(Utd)) / P(Vt)) - exp(-(V(v) + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (V(v) + P(Utd)) * exp(-(V(v) + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (V(v) + P(Utd) - P(E))) + atan(P(D) * (V(v) + P(Utd) + P(E))));

			if ((-V(v) + P(Uvm)) > 0)
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_p) + P(Ilk);
			else
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_n) - P(Ilk);

			numb imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));

			numb kv1 = (imp + Im - Id) / P(C);
			numb kx1 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-V(v) + P(Uvm)) - P(Vth_p)) * ((-V(v) + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * V(x) + P(Ds))))) * (1 - V(x)) + V(x) * (1 - 1 / (exp(P(A) * (1 - V(x)))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-V(v) + P(Uvm)) - P(Vh_n)) * ((-V(v) + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * V(x))))) * (1 - V(x)) + V(x) * (1 - 1 / (exp(P(A) * (1 - V(x)) + P(Dr)))));

			numb vmp = V(v) + 0.5f * H * kv1;
			numb xmp = V(x) + 0.5f * H * kx1;
			numb tmp = V(t) + 0.5f * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > 0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kv2 = (imp + Im - Id) / P(C);
			numb kx2 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * xmp + P(Ds))))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * xmp)))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp) + P(Dr)))));

			vmp = V(v) + 0.5f * H * kv2;
			xmp = V(x) + 0.5f * H * kx2;
			tmp = V(t) + 0.5f * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > 0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kv3 = (imp + Im - Id) / P(C);
			numb kx3 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * xmp + P(Ds))))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * xmp)))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp) + P(Dr)))));

			vmp = V(v) + H * kv3;
			xmp = V(x) + H * kx3;
			tmp = V(t) + H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > 0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kv4 = (Vnext(i) + Im - Id) / P(C);
			numb kx4 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * xmp + P(Ds))))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * xmp)))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp) + P(Dr)))));

			Vnext(v) = V(v) + H * (kv1 + 2.0f * kv2 + 2.0f * kv3 + kv4) / 6.0f;
			Vnext(x) = V(x) + H * (kx1 + 2.0f * kx2 + 2.0f * kx3 + kx4) / 6.0f;
			Vnext(t) = V(t) + H;
		}
	}
	ifSIGNAL(P(signal), triangle) {
		ifMETHOD(P(method), ExplicitRungeKutta4) {
			numb Im, Id;

			Id = P(Is) * (exp((V(v) + P(Utd)) / P(Vt)) - exp(-(V(v) + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (V(v) + P(Utd)) * exp(-(V(v) + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (V(v) + P(Utd) - P(E))) + atan(P(D) * (V(v) + P(Utd) + P(E))));

			if ((-V(v) + P(Uvm)) > 0)
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_p) + P(Ilk);
			else
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_n) - P(Ilk);

			numb imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));

			numb kv1 = (imp + Im - Id) / P(C);
			numb kx1 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-V(v) + P(Uvm)) - P(Vth_p)) * ((-V(v) + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * V(x) + P(Ds))))) * (1 - V(x)) + V(x) * (1 - 1 / (exp(P(A) * (1 - V(x)))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-V(v) + P(Uvm)) - P(Vh_n)) * ((-V(v) + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * V(x))))) * (1 - V(x)) + V(x) * (1 - 1 / (exp(P(A) * (1 - V(x)) + P(Dr)))));

			numb vmp = V(v) + 0.5f * H * kv1;
			numb xmp = V(x) + 0.5f * H * kx1;
			numb tmp = V(t) + 0.5f * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > 0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kv2 = (imp + Im - Id) / P(C);
			numb kx2 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * xmp + P(Ds))))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * xmp)))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp) + P(Dr)))));

			vmp = V(v) + 0.5f * H * kv2;
			xmp = V(x) + 0.5f * H * kx2;
			tmp = V(t) + 0.5f * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > 0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kv3 = (imp + Im - Id) / P(C);
			numb kx3 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * xmp + P(Ds))))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * xmp)))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp) + P(Dr)))));

			vmp = V(v) + H * kv3;
			xmp = V(x) + H * kx3;
			tmp = V(t) + H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > 0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kv4 = (Vnext(i) + Im - Id) / P(C);
			numb kx4 = (1 / P(tau_s)) * (1 / (1 + exp(-1 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * ((1 - 1 / (exp((P(A) * xmp + P(Ds))))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp))))) - (1 / P(tau_r)) * (1 - 1 / (1 + exp(-1 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * ((1 - 1 / (exp((P(A) * xmp)))) * (1 - xmp) + xmp * (1 - 1 / (exp(P(A) * (1 - xmp) + P(Dr)))));

			Vnext(v) = V(v) + H * (kv1 + 2.0f * kv2 + 2.0f * kv3 + kv4) / 6.0f;
			Vnext(x) = V(x) + H * (kx1 + 2.0f * kx2 + 2.0f * kx3 + kx4) / 6.0f;
			Vnext(t) = V(t) + H;
		}
	}
}