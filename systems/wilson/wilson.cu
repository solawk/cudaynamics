#include "wilson.h"

#define name wilson

namespace attributes
{
    enum variables { v, r, i, t };
    enum parameters { C, tau, p0, p1, p2, p3, p4, p5, p6, p7, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, method, COUNT };
    enum waveforms { square, sine, triangle };
    enum methods { ExplicitEuler, SemiExplicitEuler, ImplicitEuler, ExplicitMidpoint, ImplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
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
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
		}
		ifMETHOD(P(method), SemiExplicitEuler)
		{
			Vnext(t) = V(t) + H;
			Vnext(i) = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * Vnext(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
		}
		ifMETHOD(P(method), ImplicitEuler)
		{
			Vnext(i) = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			Vnext(t) = V(t) + H;
			numb v_guess = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			numb r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));

			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));

			Vnext(v) = v_guess;
			Vnext(r) = r_guess;
		}

		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			numb tmp = V(t) + H * (numb)0.5;
			numb vmp = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
			numb rmp = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			Vnext(i) = P(Idc) + (fmod((tmp - P(Idel)) > (numb)0.0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			Vnext(t) = V(t) + H;
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7)));
		}
		ifMETHOD(P(method), ImplicitMidpoint)
		{
			numb imp = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			numb tmp = V(t) + (numb)0.5 * H;
			numb v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
			numb r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));

			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));

			Vnext(t) = V(t) + H;
			Vnext(i) = P(Idc) + (fmod((tmp - P(Idel)) > (numb)0.0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
		}
		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb i1 = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			numb kt1 = V(t) + (numb)0.5 * H;

			numb kv1 = (-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + i1) / P(C);
			numb kr1 = ((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7));

			numb vmp = V(v) + (numb)0.5 * H * kv1;
			numb rmp = V(r) + (numb)0.5 * H * kr1;

			numb i2 = P(Idc) + (fmod((kt1 - P(Idel)) > (numb)0.0 ? (kt1 - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - kt1), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			numb kv2 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr2 = ((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7));

			vmp = V(v) + (numb)0.5 * H * kv2;
			rmp = V(r) + (numb)0.5 * H * kr2;

			numb kv3 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr3 = ((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7));
			Vnext(t) = V(t) + H;

			vmp = V(v) + H * kv3;
			rmp = V(r) + H * kr3;

			numb i3 = P(Idc) + (fmod((Vnext(t) - P(Idel)) > (numb)0.0 ? (Vnext(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - Vnext(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			numb kv4 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i3) / P(C);
			numb kr4 = ((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7));

			Vnext(i) = i3;
			Vnext(v) = V(v) + H * (kv1 + (numb)2.0 * kv2 + (numb)2.0 * kv3 + kv4) / (numb)6.0;
			Vnext(r) = V(r) + H * (kr1 + (numb)2.0 * kr2 + (numb)2.0 * kr3 + kr4) / (numb)6.0;
		}
		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = (numb)0.5 * H - P(symmetry);
			numb h2 = (numb)0.5 * H + P(symmetry);

			Vnext(i) = P(Idc) + (fmod((V(t) - P(Idel)) > (numb)0.0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
			Vnext(v) = V(v) + h1 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + h1 * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * Vnext(v) + P(p7)));


			numb vmp = Vnext(v);
			numb rmp = Vnext(r);

			Vnext(r) = (rmp + (h2 / P(tau)) * (P(p6) * vmp + P(p7))) / ((numb)1.0 + h2 / P(tau));

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
			Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
			Vnext(t) = V(t) + H;
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
		}
		ifMETHOD(P(method), SemiExplicitEuler)
		{
			Vnext(t) = V(t) + H;
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
			Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (Vnext(t) - P(Idel)));
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * Vnext(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
		}
		ifMETHOD(P(method), ImplicitEuler)
		{
			Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
			Vnext(t) = V(t) + H;
			numb v_guess = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			numb r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));

			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));

			Vnext(v) = v_guess;
			Vnext(r) = r_guess;
		}
		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
			numb tmp = V(t) + H * (numb)0.5;
			numb vmp = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
			numb rmp = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (tmp - P(Idel)));
			Vnext(t) = V(t) + H;
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7)));
		}
		ifMETHOD(P(method), ImplicitMidpoint)
		{
			numb imp = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
			numb tmp = V(t) + (numb)0.5 * H;
			numb v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
			numb r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));

			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));

			Vnext(t) = V(t) + H;
			Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (tmp - P(Idel)));
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
		}
		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb i1 = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
			numb kt1 = V(t) + (numb)0.5 * H;

			numb kv1 = (-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + i1) / P(C);
			numb kr1 = ((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7));

			numb vmp = V(v) + (numb)0.5 * H * kv1;
			numb rmp = V(r) + (numb)0.5 * H * kr1;

			numb i2 = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (kt1 - P(Idel)));
			numb kv2 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr2 = ((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7));

			vmp = V(v) + (numb)0.5 * H * kv2;
			rmp = V(r) + (numb)0.5 * H * kr2;

			numb kv3 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr3 = ((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7));
			Vnext(t) = V(t) + H;

			vmp = V(v) + H * kv3;
			rmp = V(r) + H * kr3;

			numb i3 = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (Vnext(t) - P(Idel)));
			numb kv4 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i3) / P(C);
			numb kr4 = ((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7));

			Vnext(i) = i3;
			Vnext(v) = V(v) + H * (kv1 + (numb)2.0 * kv2 + (numb)2.0 * kv3 + kv4) / (numb)6.0;
			Vnext(r) = V(r) + H * (kr1 + (numb)2.0 * kr2 + (numb)2.0 * kr3 + kr4) / (numb)6.0;
		}
		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = (numb)0.5 * H - P(symmetry);
			numb h2 = (numb)0.5 * H + P(symmetry);

			Vnext(i) = P(Idc) + P(Iamp) * sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
			Vnext(v) = V(v) + h1 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + h1 * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * Vnext(v) + P(p7)));


			numb vmp = Vnext(v);
			numb rmp = Vnext(r);

			Vnext(r) = (rmp + (h2 / P(tau)) * (P(p6) * vmp + P(p7))) / ((numb)1.0 + h2 / P(tau));

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
			Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
			Vnext(t) = V(t) + H;
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
		}
		ifMETHOD(P(method), SemiExplicitEuler)
		{
			Vnext(t) = V(t) + H;
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));
			Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (Vnext(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (Vnext(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (Vnext(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * Vnext(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
		}
		ifMETHOD(P(method), ImplicitEuler)
		{
			Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
			Vnext(t) = V(t) + H;
			numb v_guess = V(v) + H * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			numb r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));
			v_guess = V(v) + H * ((-(P(p0) + P(p1) * v_guess + P(p2) * v_guess * v_guess) * (v_guess - P(p3)) - P(p5) * r_guess * (v_guess - P(p4)) + Vnext(i)) / P(C));

			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));
			r_guess = V(r) + H * (((numb)1.0 / P(tau)) * (-r_guess + P(p6) * v_guess + P(p7)));

			Vnext(v) = v_guess;
			Vnext(r) = r_guess;
		}
		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
			numb tmp = V(t) + H * (numb)0.5;
			numb vmp = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
			numb rmp = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0))));
			Vnext(t) = V(t) + H;
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7)));
		}
		ifMETHOD(P(method), ImplicitMidpoint)
		{
			numb imp = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
			numb tmp = V(t) + (numb)0.5 * H;
			numb v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + imp) / P(C));
			numb r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7)));

			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));
			v_mid_guess = V(v) + H * (numb)0.5 * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + imp) / P(C));

			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
			r_mid_guess = V(r) + H * (numb)0.5 * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));

			Vnext(t) = V(t) + H;
			Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0))));
			Vnext(v) = V(v) + H * ((-(P(p0) + P(p1) * v_mid_guess + P(p2) * v_mid_guess * v_mid_guess) * (v_mid_guess - P(p3)) - P(p5) * r_mid_guess * (v_mid_guess - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + H * (((numb)1.0 / P(tau)) * (-r_mid_guess + P(p6) * v_mid_guess + P(p7)));
		}
		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb i1 = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
			numb kt1 = V(t) + (numb)0.5 * H;

			numb kv1 = (-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + i1) / P(C);
			numb kr1 = ((numb)1.0 / P(tau)) * (-V(r) + P(p6) * V(v) + P(p7));

			numb vmp = V(v) + (numb)0.5 * H * kv1;
			numb rmp = V(r) + (numb)0.5 * H * kr1;

			numb i2 = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (kt1 - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (kt1 - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (kt1 - P(Idel)) + (numb)1.0) / (numb)2.0))));
			numb kv2 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr2 = ((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7));

			vmp = V(v) + (numb)0.5 * H * kv2;
			rmp = V(r) + (numb)0.5 * H * kr2;

			numb kv3 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i2) / P(C);
			numb kr3 = ((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7));
			Vnext(t) = V(t) + H;

			vmp = V(v) + H * kv3;
			rmp = V(r) + H * kr3;

			numb i3 = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (Vnext(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (Vnext(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (Vnext(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
			numb kv4 = (-(P(p0) + P(p1) * vmp + P(p2) * vmp * vmp) * (vmp - P(p3)) - P(p5) * rmp * (vmp - P(p4)) + i3) / P(C);
			numb kr4 = ((numb)1.0 / P(tau)) * (-rmp + P(p6) * vmp + P(p7));

			Vnext(t) = V(t) + H;
			Vnext(i) = i3;
			Vnext(v) = V(v) + H * (kv1 + (numb)2.0 * kv2 + (numb)2.0 * kv3 + kv4) / (numb)6.0;
			Vnext(r) = V(r) + H * (kr1 + (numb)2.0 * kr2 + (numb)2.0 * kr3 + kr4) / (numb)6.0;

		}
		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = (numb)0.5 * H - P(symmetry);
			numb h2 = (numb)0.5 * H + P(symmetry);

			Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0))));
			Vnext(v) = V(v) + h1 * ((-(P(p0) + P(p1) * V(v) + P(p2) * V(v) * V(v)) * (V(v) - P(p3)) - P(p5) * V(r) * (V(v) - P(p4)) + Vnext(i)) / P(C));
			Vnext(r) = V(r) + h1 * (((numb)1.0 / P(tau)) * (-V(r) + P(p6) * Vnext(v) + P(p7)));

			numb vmp = Vnext(v);
			numb rmp = Vnext(r);

			Vnext(r) = (rmp + (h2 / P(tau)) * (P(p6) * vmp + P(p7))) / ((numb)1.0 + h2 / P(tau));

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