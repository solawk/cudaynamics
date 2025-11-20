#include "main.h"
#include "hodgkin_huxley.h"

#define name hodgkin_huxley

namespace attributes
{
	enum variables { n, m, h, v, i, t };
	enum parameters { G_Na, G_leak, G_K, E_Na, E_leak, E_K, C, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, method, COUNT };
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
			Vnext(i) = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			Vnext(t) = V(t) + H;

			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			Vnext(n) = V(n) + H * (alpha_n * (1 - V(n)) - beta_n * V(n));
			Vnext(m) = V(m) + H * (alpha_m * (1 - V(m)) - beta_m * V(m));
			Vnext(h) = V(h) + H * (alpha_h * (1 - V(h)) - beta_h * V(h));

			numb I_Na = (V(m) * V(m) * V(m)) * P(G_Na) * V(h) * (V(v) - P(E_Na));
			numb I_K = (V(n) * V(n) * V(n) * V(n)) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			Vnext(v) = V(v) + H * ((Vnext(i) - I_K - I_Na - I_L) / P(C));
		}

		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			numb tmp = V(t) + H * 0.5;

			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			numb nmp = V(n) + H * 0.5 * (alpha_n * (1 - V(n)) - beta_n * V(n));
			numb mmp = V(m) + H * 0.5 * (alpha_m * (1 - V(m)) - beta_m * V(m));
			numb hmp = V(h) + H * 0.5 * (alpha_h * (1 - V(h)) - beta_h * V(h));

			numb I_Na = (V(m) * V(m) * V(m)) * P(G_Na) * V(h) * (V(v) - P(E_Na));
			numb I_K = (V(n) * V(n) * V(n) * V(n)) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			numb vmp = V(v) + H * 0.5 * ((imp - I_K - I_Na - I_L) / P(C));

			Vnext(i) = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			Vnext(t) = V(t) + H;

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			Vnext(n) = V(n) + H * (alpha_n * (1 - nmp) - beta_n * nmp);
			Vnext(m) = V(m) + H * (alpha_m * (1 - mmp) - beta_m * mmp);
			Vnext(h) = V(h) + H * (alpha_h * (1 - hmp) - beta_h * hmp);

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			Vnext(v) = V(v) + H * ((Vnext(i) - I_K - I_Na - I_L) / P(C));
		}

		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb imp = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			numb tmp = V(t) + H * 0.5f;

			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			numb kn1 = alpha_n * (1 - V(n)) - beta_n * V(n);
			numb km1 = alpha_m * (1 - V(m)) - beta_m * V(m);
			numb kh1 = alpha_h * (1 - V(h)) - beta_h * V(h);

			numb I_Na = (V(m) * V(m) * V(m)) * P(G_Na) * V(h) * (V(v) - P(E_Na));
			numb I_K = (V(n) * V(n) * V(n) * V(n)) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			numb kv1 = (imp - I_K - I_Na - I_L) / P(C);

			numb nmp = V(n) + H * 0.5f * kn1;
			numb mmp = V(m) + H * 0.5f * km1;
			numb hmp = V(h) + H * 0.5f * kh1;
			numb vmp = V(v) + H * 0.5f * kv1;

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn2 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km2 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh2 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			numb kv2 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * 0.5f * kn2;
			mmp = V(m) + H * 0.5f * km2;
			hmp = V(h) + H * 0.5f * kh2;
			vmp = V(v) + H * 0.5f * kv2;

			Vnext(t) = V(t) + H;

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn3 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km3 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh3 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			numb kv3 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * kn3;
			mmp = V(m) + H * km3;
			hmp = V(h) + H * kh3;
			vmp = V(v) + H * kv3;

			Vnext(i) = P(Idc) + (fmodf((Vnext(t) - P(Idel)) > 0 ? (Vnext(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - Vnext(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn4 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km4 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh4 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			numb kv4 = (Vnext(i) - I_K - I_Na - I_L) / P(C);

			Vnext(n) = V(n) + H * (kn1 + 2.0f * kn2 + 2.0f * kn3 + kn4) / 6.0f;
			Vnext(m) = V(m) + H * (km1 + 2.0f * km2 + 2.0f * km3 + km4) / 6.0f;
			Vnext(h) = V(h) + H * (kh1 + 2.0f * kh2 + 2.0f * kh3 + kh4) / 6.0f;
			Vnext(v) = V(v) + H * (kv1 + 2.0f * kv2 + 2.0f * kv3 + kv4) / 6.0f;
		}

		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = 0.5 * H - P(symmetry);
			numb h2 = 0.5 * H + P(symmetry);

			numb tmp = V(t) + h1;
			Vnext(i) = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			numb nmp = V(n) + h1 * (alpha_n * (1 - V(n)) - beta_n * V(n));
			numb mmp = V(m) + h1 * (alpha_m * (1 - V(m)) - beta_m * V(m));
			numb hmp = V(h) + h1 * (alpha_h * (1 - V(h)) - beta_h * V(h));

			numb I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (V(v) - P(E_Na));
			numb I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			numb vmp = V(v) + h1 * ((Vnext(i) - I_K - I_Na - I_L) / P(C));

			I_L = P(G_leak) * (vmp - P(E_leak));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));

			Vnext(v) = vmp + h2 * ((Vnext(i) - I_K - I_Na - I_L) / P(C));

			alpha_h = 0.07 * exp(-Vnext(v) / 20);
			beta_h = 1 / (exp((30 - Vnext(v)) / 10) + 1);
			alpha_m = 0.1 * ((25 - Vnext(v)) / (exp((25 - Vnext(v)) / 10) - 1));
			beta_m = 4 * exp(-Vnext(v) / 18);
			alpha_n = 0.01 * ((10 - Vnext(v)) / (exp((10 - Vnext(v)) / 10) - 1));
			beta_n = 0.125 * exp(-Vnext(v) / 80);

			Vnext(h) = (hmp + h2 * alpha_h) / (1 + h2 * (alpha_h + beta_h));
			Vnext(m) = (mmp + h2 * alpha_m) / (1 + h2 * (alpha_m + beta_m));
			Vnext(n) = (nmp + h2 * alpha_n) / (1 + h2 * (alpha_n + beta_n));
			Vnext(t) = V(t) + H;
		}
	}

	ifSIGNAL(P(signal), sine)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
			Vnext(t) = V(t) + H;

			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			Vnext(n) = V(n) + H * (alpha_n * (1 - V(n)) - beta_n * V(n));
			Vnext(m) = V(m) + H * (alpha_m * (1 - V(m)) - beta_m * V(m));
			Vnext(h) = V(h) + H * (alpha_h * (1 - V(h)) - beta_h * V(h));

			numb I_Na = (V(m) * V(m) * V(m)) * P(G_Na) * V(h) * (V(v) - P(E_Na));
			numb I_K = (V(n) * V(n) * V(n) * V(n)) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			Vnext(v) = V(v) + H * ((Vnext(i) - I_K - I_Na - I_L) / P(C));
		}

		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
			numb tmp = V(t) + H * 0.5;

			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			numb nmp = V(n) + H * 0.5 * (alpha_n * (1 - V(n)) - beta_n * V(n));
			numb mmp = V(m) + H * 0.5 * (alpha_m * (1 - V(m)) - beta_m * V(m));
			numb hmp = V(h) + H * 0.5 * (alpha_h * (1 - V(h)) - beta_h * V(h));

			numb I_Na = (V(m) * V(m) * V(m)) * P(G_Na) * V(h) * (V(v) - P(E_Na));
			numb I_K = (V(n) * V(n) * V(n) * V(n)) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			numb vmp = V(v) + H * 0.5 * ((imp - I_K - I_Na - I_L) / P(C));

			Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));
			Vnext(t) = V(t) + H;

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			Vnext(n) = V(n) + H * (alpha_n * (1 - nmp) - beta_n * nmp);
			Vnext(m) = V(m) + H * (alpha_m * (1 - mmp) - beta_m * mmp);
			Vnext(h) = V(h) + H * (alpha_h * (1 - hmp) - beta_h * hmp);

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			Vnext(v) = V(v) + H * ((Vnext(i) - I_K - I_Na - I_L) / P(C));
		}

		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));
			numb tmp = V(t) + 0.5f * H;

			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			numb kn1 = alpha_n * (1 - V(n)) - beta_n * V(n);
			numb km1 = alpha_m * (1 - V(m)) - beta_m * V(m);
			numb kh1 = alpha_h * (1 - V(h)) - beta_h * V(h);

			numb I_Na = (V(m) * V(m) * V(m)) * P(G_Na) * V(h) * (V(v) - P(E_Na));
			numb I_K = (V(n) * V(n) * V(n) * V(n)) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			numb kv1 = (imp - I_K - I_Na - I_L) / P(C);

			numb nmp = V(n) + H * 0.5f * kn1;
			numb mmp = V(m) + H * 0.5f * km1;
			numb hmp = V(h) + H * 0.5f * kh1;
			numb vmp = V(v) + H * 0.5f * kv1;

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn2 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km2 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh2 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			numb kv2 = (imp - I_K - I_Na - I_L) / P(C);
			nmp = V(n) + H * 0.5f * kn2;
			mmp = V(m) + H * 0.5f * km2;
			hmp = V(h) + H * 0.5f * kh2;
			vmp = V(v) + H * 0.5f * kv2;

			Vnext(t) = V(t) + H;

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn3 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km3 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh3 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			numb kv3 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * kn3;
			mmp = V(m) + H * km3;
			hmp = V(h) + H * kh3;
			vmp = V(v) + H * kv3;

			Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (Vnext(t) - P(Idel)));

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn4 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km4 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh4 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			numb kv4 = (Vnext(i) - I_K - I_Na - I_L) / P(C);

			Vnext(n) = V(n) + H * (kn1 + 2.0f * kn2 + 2.0f * kn3 + kn4) / 6.0f;
			Vnext(m) = V(m) + H * (km1 + 2.0f * km2 + 2.0f * km3 + km4) / 6.0f;
			Vnext(h) = V(h) + H * (kh1 + 2.0f * kh2 + 2.0f * kh3 + kh4) / 6.0f;
			Vnext(v) = V(v) + H * (kv1 + 2.0f * kv2 + 2.0f * kv3 + kv4) / 6.0f;
		}

		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = 0.5 * H - P(symmetry);
			numb h2 = 0.5 * H + P(symmetry);

			numb tmp = V(t) + h1;
			Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			numb nmp = V(n) + h1 * (alpha_n * (1 - V(n)) - beta_n * V(n));
			numb mmp = V(m) + h1 * (alpha_m * (1 - V(m)) - beta_m * V(m));
			numb hmp = V(h) + h1 * (alpha_h * (1 - V(h)) - beta_h * V(h));

			numb I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (V(v) - P(E_Na));
			numb I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			numb vmp = V(v) + h1 * ((Vnext(i) - I_K - I_Na - I_L) / P(C));

			I_L = P(G_leak) * (vmp - P(E_leak));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));

			Vnext(v) = vmp + h2 * ((Vnext(i) - I_K - I_Na - I_L) / P(C));

			alpha_h = 0.07 * exp(-Vnext(v) / 20);
			beta_h = 1 / (exp((30 - Vnext(v)) / 10) + 1);
			alpha_m = 0.1 * ((25 - Vnext(v)) / (exp((25 - Vnext(v)) / 10) - 1));
			beta_m = 4 * exp(-Vnext(v) / 18);
			alpha_n = 0.01 * ((10 - Vnext(v)) / (exp((10 - Vnext(v)) / 10) - 1));
			beta_n = 0.125 * exp(-Vnext(v) / 80);

			Vnext(h) = (hmp + h2 * alpha_h) / (1 + h2 * (alpha_h + beta_h));
			Vnext(m) = (mmp + h2 * alpha_m) / (1 + h2 * (alpha_m + beta_m));
			Vnext(n) = (nmp + h2 * alpha_n) / (1 + h2 * (alpha_n + beta_n));
			Vnext(t) = V(t) + H;
		}
	}

	ifSIGNAL(P(signal), triangle)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
			Vnext(t) = V(t) + H;
			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			Vnext(n) = V(n) + H * (alpha_n * (1 - V(n)) - beta_n * V(n));
			Vnext(m) = V(m) + H * (alpha_m * (1 - V(m)) - beta_m * V(m));
			Vnext(h) = V(h) + H * (alpha_h * (1 - V(h)) - beta_h * V(h));

			numb I_Na = (V(m) * V(m) * V(m)) * P(G_Na) * V(h) * (V(v) - P(E_Na));
			numb I_K = (V(n) * V(n) * V(n) * V(n)) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			Vnext(v) = V(v) + H * ((Vnext(i) - I_K - I_Na - I_L) / P(C));
		}

		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
			numb tmp = V(t) + H * 0.5f;

			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			numb nmp = V(n) + H * 0.5 * (alpha_n * (1 - V(n)) - beta_n * V(n));
			numb mmp = V(m) + H * 0.5 * (alpha_m * (1 - V(m)) - beta_m * V(m));
			numb hmp = V(h) + H * 0.5 * (alpha_h * (1 - V(h)) - beta_h * V(h));

			numb I_Na = (V(m) * V(m) * V(m)) * P(G_Na) * V(h) * (V(v) - P(E_Na));
			numb I_K = (V(n) * V(n) * V(n) * V(n)) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			numb vmp = V(v) + H * 0.5 * ((imp - I_K - I_Na - I_L) / P(C));

			Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));
			Vnext(t) = V(t) + H;

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			Vnext(n) = V(n) + H * (alpha_n * (1 - nmp) - beta_n * nmp);
			Vnext(m) = V(m) + H * (alpha_m * (1 - mmp) - beta_m * mmp);
			Vnext(h) = V(h) + H * (alpha_h * (1 - hmp) - beta_h * hmp);

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			Vnext(v) = V(v) + H * ((Vnext(i) - I_K - I_Na - I_L) / P(C));
		}

		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));
			numb tmp = V(t) + 0.5f * H;

			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			numb kn1 = alpha_n * (1 - V(n)) - beta_n * V(n);
			numb km1 = alpha_m * (1 - V(m)) - beta_m * V(m);
			numb kh1 = alpha_h * (1 - V(h)) - beta_h * V(h);

			numb I_Na = (V(m) * V(m) * V(m)) * P(G_Na) * V(h) * (V(v) - P(E_Na));
			numb I_K = (V(n) * V(n) * V(n) * V(n)) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			numb kv1 = (imp - I_K - I_Na - I_L) / P(C);

			numb nmp = V(n) + H * 0.5f * kn1;
			numb mmp = V(m) + H * 0.5f * km1;
			numb hmp = V(h) + H * 0.5f * kh1;
			numb vmp = V(v) + H * 0.5f * kv1;

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn2 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km2 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh2 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			numb kv2 = (imp - I_K - I_Na - I_L) / P(C);
			nmp = V(n) + H * 0.5f * kn2;
			mmp = V(m) + H * 0.5f * km2;
			hmp = V(h) + H * 0.5f * kh2;
			vmp = V(v) + H * 0.5f * kv2;

			Vnext(t) = V(t) + H;

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn3 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km3 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh3 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			numb kv3 = (imp - I_K - I_Na - I_L) / P(C);
			nmp = V(n) + H * kn3;
			mmp = V(m) + H * km3;
			hmp = V(h) + H * kh3;
			vmp = V(v) + H * kv3;
			
			Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (Vnext(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (Vnext(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (Vnext(t) - P(Idel)) + 1.0f) / 2.0f)));

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn4 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km4 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh4 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			numb kv4 = (Vnext(i) - I_K - I_Na - I_L) / P(C);

			Vnext(n) = V(n) + H * (kn1 + 2.0f * kn2 + 2.0f * kn3 + kn4) / 6.0f;
			Vnext(m) = V(m) + H * (km1 + 2.0f * km2 + 2.0f * km3 + km4) / 6.0f;
			Vnext(h) = V(h) + H * (kh1 + 2.0f * kh2 + 2.0f * kh3 + kh4) / 6.0f;
			Vnext(v) = V(v) + H * (kv1 + 2.0f * kv2 + 2.0f * kv3 + kv4) / 6.0f;

		}

		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = 0.5 * H - P(symmetry);
			numb h2 = 0.5 * H + P(symmetry);

			numb tmp = V(t) + h1;
			Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb alpha_n = 0.01 * ((10 - V(v)) / (exp((10 - V(v)) / 10) - 1));
			numb beta_n = 0.125 * exp(-V(v) / 80);
			numb alpha_m = 0.1 * ((25 - V(v)) / (exp((25 - V(v)) / 10) - 1));
			numb beta_m = 4 * exp(-V(v) / 18);
			numb alpha_h = 0.07 * exp(-V(v) / 20);
			numb beta_h = 1 / (exp((30 - V(v)) / 10) + 1);

			numb nmp = V(n) + h1 * (alpha_n * (1 - V(n)) - beta_n * V(n));
			numb mmp = V(m) + h1 * (alpha_m * (1 - V(m)) - beta_m * V(m));
			numb hmp = V(h) + h1 * (alpha_h * (1 - V(h)) - beta_h * V(h));

			numb I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (V(v) - P(E_Na));
			numb I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (V(v) - P(E_K));
			numb I_L = P(G_leak) * (V(v) - P(E_leak));

			numb vmp = V(v) + h1 * ((Vnext(i) - I_K - I_Na - I_L) / P(C));

			I_L = P(G_leak) * (vmp - P(E_leak));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));

			Vnext(v) = vmp + h2 * ((Vnext(i) - I_K - I_Na - I_L) / P(C));

			alpha_h = 0.07 * exp(-Vnext(v) / 20);
			beta_h = 1 / (exp((30 - Vnext(v)) / 10) + 1);
			alpha_m = 0.1 * ((25 - Vnext(v)) / (exp((25 - Vnext(v)) / 10) - 1));
			beta_m = 4 * exp(-Vnext(v) / 18);
			alpha_n = 0.01 * ((10 - Vnext(v)) / (exp((10 - Vnext(v)) / 10) - 1));
			beta_n = 0.125 * exp(-Vnext(v) / 80);

			Vnext(h) = (hmp + h2 * alpha_h) / (1 + h2 * (alpha_h + beta_h));
			Vnext(m) = (mmp + h2 * alpha_m) / (1 + h2 * (alpha_m + beta_m));
			Vnext(n) = (nmp + h2 * alpha_n) / (1 + h2 * (alpha_n + beta_n));
			Vnext(t) = V(t) + H;
		}
	}
}

#undef name