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
	ifSIGNAL(P(signal), square) {
		ifMETHOD(P(method), ExplicitRungeKutta4) {
			numb Im, Id;

			Id = P(Is) * (exp((V(v) + P(Utd)) / P(Vt)) - exp(-(V(v) + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (V(v) + P(Utd)) * exp(-(V(v) + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (V(v) + P(Utd) - P(E))) + atan(P(D) * (V(v) + P(Utd) + P(E))));

			if ((-V(v) + P(Uvm)) > (numb)0.0)
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_p) + P(Ilk);
			else
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_n) - P(Ilk);

			numb imp = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);

			numb kv1 = (imp + Im - Id) / P(C);
			numb kx1 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-V(v) + P(Uvm)) - P(Vth_p)) * ((-V(v) + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * V(x) + P(Ds))))) * ((numb)1.0 - V(x)) + V(x) * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - V(x)))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-V(v) + P(Uvm)) - P(Vh_n)) * ((-V(v) + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * V(x))))) * ((numb)1.0 - V(x)) + V(x) * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - V(x)) + P(Dr)))));

			numb vmp = V(v) + (numb)0.5 * H * kv1;
			numb xmp = V(x) + (numb)0.5 * H * kx1;
			numb tmp = V(t) + (numb)0.5 * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > (numb)0.0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + (fmod((tmp - P(Idel)) > (numb)0.0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);

			numb kv2 = (imp + Im - Id) / P(C);
			numb kx2 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp + P(Ds))))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp)))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp) + P(Dr)))));

			vmp = V(v) + (numb)0.5 * H * kv2;
			xmp = V(x) + (numb)0.5 * H * kx2;
			tmp = V(t) + (numb)0.5 * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > (numb)0.0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + (fmod((tmp - P(Idel)) > (numb)0.0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);

			numb kv3 = (imp + Im - Id) / P(C);
			numb kx3 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp + P(Ds))))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp)))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp) + P(Dr)))));

			vmp = V(v) + H * kv3;
			xmp = V(x) + H * kx3;
			tmp = V(t) + H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > (numb)0.0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			Vnext(i) = P(Idc) + (fmod((tmp - P(Idel)) > (numb)0.0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);

			numb kv4 = (Vnext(i) + Im - Id) / P(C);
			numb kx4 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp + P(Ds))))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp)))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp) + P(Dr)))));

			Vnext(v) = V(v) + H * (kv1 + (numb)2.0 * kv2 + (numb)2.0 * kv3 + kv4) / (numb)6.0;
			Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
			Vnext(t) = V(t) + H;
		}
	}
	ifSIGNAL(P(signal), sine) {
		ifMETHOD(P(method), ExplicitRungeKutta4) {
			numb Im, Id;

			Id = P(Is) * (exp((V(v) + P(Utd)) / P(Vt)) - exp(-(V(v) + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (V(v) + P(Utd)) * exp(-(V(v) + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (V(v) + P(Utd) - P(E))) + atan(P(D) * (V(v) + P(Utd) + P(E))));

			if ((-V(v) + P(Uvm)) > (numb)0.0)
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_p) + P(Ilk);
			else
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_n) - P(Ilk);

			numb imp = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));

			numb kv1 = (imp + Im - Id) / P(C);
			numb kx1 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-V(v) + P(Uvm)) - P(Vth_p)) * ((-V(v) + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * V(x) + P(Ds))))) * ((numb)1.0 - V(x)) + V(x) * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - V(x)))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-V(v) + P(Uvm)) - P(Vh_n)) * ((-V(v) + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * V(x))))) * ((numb)1.0 - V(x)) + V(x) * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - V(x)) + P(Dr)))));

			numb vmp = V(v) + (numb)0.5 * H * kv1;
			numb xmp = V(x) + (numb)0.5 * H * kx1;
			numb tmp = V(t) + (numb)0.5 * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > (numb)0.0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (tmp - P(Idel)));

			numb kv2 = (imp + Im - Id) / P(C);
			numb kx2 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp + P(Ds))))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp)))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp) + P(Dr)))));

			vmp = V(v) + (numb)0.5 * H * kv2;
			xmp = V(x) + (numb)0.5 * H * kx2;
			tmp = V(t) + (numb)0.5 * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > (numb)0.0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (tmp - P(Idel)));

			numb kv3 = (imp + Im - Id) / P(C);
			numb kx3 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp + P(Ds))))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp)))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp) + P(Dr)))));

			vmp = V(v) + H * kv3;
			xmp = V(x) + H * kx3;
			tmp = V(t) + H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > (numb)0.0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (tmp - P(Idel)));

			numb kv4 = (Vnext(i) + Im - Id) / P(C);
			numb kx4 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp + P(Ds))))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp)))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp) + P(Dr)))));

			Vnext(v) = V(v) + H * (kv1 + (numb)2.0 * kv2 + (numb)2.0 * kv3 + kv4) / (numb)6.0;
			Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
			Vnext(t) = V(t) + H;
		}
	}
	ifSIGNAL(P(signal), triangle) {
		ifMETHOD(P(method), ExplicitRungeKutta4) {
			numb Im, Id;

			Id = P(Is) * (exp((V(v) + P(Utd)) / P(Vt)) - exp(-(V(v) + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (V(v) + P(Utd)) * exp(-(V(v) + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (V(v) + P(Utd) - P(E))) + atan(P(D) * (V(v) + P(Utd) + P(E))));

			if ((-V(v) + P(Uvm)) > (numb)0.0)
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_p) + P(Ilk);
			else
				Im = (-V(v) + P(Uvm)) * V(x) / P(Ron_n) - P(Ilk);

			numb imp = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));

			numb kv1 = (imp + Im - Id) / P(C);
			numb kx1 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-V(v) + P(Uvm)) - P(Vth_p)) * ((-V(v) + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * V(x) + P(Ds))))) * ((numb)1.0 - V(x)) + V(x) * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - V(x)))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-V(v) + P(Uvm)) - P(Vh_n)) * ((-V(v) + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * V(x))))) * ((numb)1.0 - V(x)) + V(x) * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - V(x)) + P(Dr)))));

			numb vmp = V(v) + (numb)0.5 * H * kv1;
			numb xmp = V(x) + (numb)0.5 * H * kx1;
			numb tmp = V(t) + (numb)0.5 * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > (numb)0.0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0))));

			numb kv2 = (imp + Im - Id) / P(C);
			numb kx2 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp + P(Ds))))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp)))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp) + P(Dr)))));

			vmp = V(v) + (numb)0.5 * H * kv2;
			xmp = V(x) + (numb)0.5 * H * kx2;
			tmp = V(t) + (numb)0.5 * H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > (numb)0.0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			imp = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0))));

			numb kv3 = (imp + Im - Id) / P(C);
			numb kx3 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp + P(Ds))))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp)))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp) + P(Dr)))));

			vmp = V(v) + H * kv3;
			xmp = V(x) + H * kx3;
			tmp = V(t) + H;


			Id = P(Is) * (exp((vmp + P(Utd)) / P(Vt)) - exp(-(vmp + P(Utd)) / P(Vt))) + (P(Ip) / P(Vp)) * (vmp + P(Utd)) * exp(-(vmp + P(Utd) - P(Vp)) / P(Vp)) + P(Iv) * (atan(P(D) * (vmp + P(Utd) - P(E))) + atan(P(D) * (vmp + P(Utd) + P(E))));

			if ((-vmp + P(Uvm)) > (numb)0.0)
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_p) + P(Ilk);
			else
				Im = (-vmp + P(Uvm)) * xmp / P(Ron_n) - P(Ilk);

			Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0))));

			numb kv4 = (Vnext(i) + Im - Id) / P(C);
			numb kx4 = ((numb)1.0 / P(tau_s)) * ((numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vs) * P(Vs)) * ((-vmp + P(Uvm)) - P(Vth_p)) * ((-vmp + P(Uvm)) - P(Vth_n))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp + P(Ds))))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp))))) - ((numb)1.0 / P(tau_r)) * ((numb)1.0 - (numb)1.0 / ((numb)1.0 + exp(-(numb)1.0 / (P(Vr) * P(Vr)) * ((-vmp + P(Uvm)) - P(Vh_n)) * ((-vmp + P(Uvm)) - P(Vh_p))))) * (((numb)1.0 - (numb)1.0 / (exp((P(A) * xmp)))) * ((numb)1.0 - xmp) + xmp * ((numb)1.0 - (numb)1.0 / (exp(P(A) * ((numb)1.0 - xmp) + P(Dr)))));

			Vnext(v) = V(v) + H * (kv1 + (numb)2.0 * kv2 + (numb)2.0 * kv3 + kv4) / (numb)6.0;
			Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
			Vnext(t) = V(t) + H;
		}
	}
}