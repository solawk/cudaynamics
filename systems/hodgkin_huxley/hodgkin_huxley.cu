#include "hodgkin_huxley.h"

#define name hodgkin_huxley

namespace attributes
{
	enum variables { n, m, h, v, i, t };
	enum parameters { G_Na, G_leak, G_K, E_Na, E_leak, E_K, C, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, method, COUNT };
	enum waveforms { square, sine, triangle };
	enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD, ExplicitDormandPrince8 };
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

		ifMETHOD(P(method), ExplicitDormandPrince8)
		{
			numb M[13][12] = {	{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
								{0.05555555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
								{0.02083333333333, 0.0625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
								{0.03125, 0, 0.09375, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
								{0.3125, 0, -1.171875, 1.171875, 0, 0, 0, 0, 0, 0, 0, 0}, 
								{0.0375, 0, 0, 0.1875, 0.15, 0, 0, 0, 0, 0, 0, 0}, 
								{0.04791013711111, 0, 0, 0.1122487127778, -0.02550567377778, 0.01284682388889, 0, 0, 0, 0, 0, 0}, 
								{0.01691798978729, 0, 0, 0.387848278486, 0.0359773698515, 0.1969702142157, -0.1727138523405, 0, 0, 0, 0, 0}, 
								{0.06909575335919, 0, 0, -0.6342479767289, -0.1611975752246, 0.1386503094588, 0.9409286140358, 0.2116363264819, 0, 0, 0, 0}, 
								{0.183556996839, 0, 0, -2.468768084316, -0.2912868878163, -0.02647302023312, 2.847838764193, 0.2813873314699, 0.1237448998633, 0, 0, 0}, 
								{-1.215424817396, 0, 0, 16.67260866595, 0.9157418284168, -6.056605804357, -16.00357359416, 14.8493030863, -13.37157573529, 5.13418264818, 0, 0}, 
								{0.2588609164383, 0, 0, -4.774485785489, -0.435093013777, -3.049483332072, 5.577920039936, 6.155831589861, -5.062104586737, 2.193926173181, 0.1346279986593, 0}, 
								{0.8224275996265, 0, 0, -11.65867325728, -0.7576221166909, 0.7139735881596, 12.07577498689, -2.12765911392, 1.990166207049, -0.234286471544, 0.1758985777079, 0} };

			numb B[2][14] = {	{0.04174749114153, 0, 0, 0, 0, -0.05545232861124, 0.2393128072012, 0.7035106694034, -0.7597596138145, 0.6605630309223, 0.1581874825101, -0.2381095387529, 0.25, 0}, 
								{0.02955321367635, 0, 0, 0, 0, -0.8286062764878, 0.3112409000511, 2.4673451906, -2.546941651842, 1.443548583677, 0.07941559588113, 0.04444444444444, 0, 0} };

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

			numb imp = P(Idc) + (fmodf((V(t) - P(Idel)) > 0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);
			
			numb kv1 = (imp - I_K - I_Na - I_L) / P(C);

			numb nmp = V(n) + H * M[1][0] * kn1;
			numb mmp = V(m) + H * M[1][0] * km1;
			numb hmp = V(h) + H * M[1][0] * kh1;
			numb vmp = V(v) + H * M[1][0] * kv1;
			numb tmp = V(t) + H * M[1][0];

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

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv2 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[2][0] * kn1 + M[2][1] * kn2);
			mmp = V(m) + H * (M[2][0] * km1 + M[2][1] * km2);
			hmp = V(h) + H * (M[2][0] * kh1 + M[2][1] * kh2);
			vmp = V(v) + H * (M[2][0] * kv1 + M[2][1] * kv2);
			tmp = V(t) + H * (M[2][0] + M[2][1]);

			
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

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv3 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[3][0] * kn1 + M[3][1] * kn2 + M[3][2] * kn3);
			mmp = V(m) + H * (M[3][0] * km1 + M[2][1] * km2 + M[3][2] * km3);
			hmp = V(h) + H * (M[3][0] * kh1 + M[2][1] * kh2 + M[3][2] * kh3);
			vmp = V(v) + H * (M[3][0] * kv1 + M[2][1] * kv2 + M[3][2] * kv3);
			tmp = V(t) + H * (M[3][0] + M[3][1] + M[3][2]);

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

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv4 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[4][0] * kn1 + M[4][1] * kn2 + M[4][2] * kn3 + M[4][3] * kn4);
			mmp = V(m) + H * (M[4][0] * km1 + M[4][1] * km2 + M[4][2] * km3 + M[4][3] * km4);
			hmp = V(h) + H * (M[4][0] * kh1 + M[4][1] * kh2 + M[4][2] * kh3 + M[4][3] * kh4);
			vmp = V(v) + H * (M[4][0] * kv1 + M[4][1] * kv2 + M[4][2] * kv3 + M[4][3] * kv4);
			tmp = V(t) + H * (M[4][0] + M[4][1] + M[4][2] + M[4][3]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn5 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km5 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh5 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv5 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[5][0] * kn1 + M[5][1] * kn2 + M[5][2] * kn3 + M[5][3] * kn4 + M[5][4] * kn5);
			mmp = V(m) + H * (M[5][0] * km1 + M[5][1] * km2 + M[5][2] * km3 + M[5][3] * km4 + M[5][4] * km5);
			hmp = V(h) + H * (M[5][0] * kh1 + M[5][1] * kh2 + M[5][2] * kh3 + M[5][3] * kh4 + M[5][4] * kh5);
			vmp = V(v) + H * (M[5][0] * kv1 + M[5][1] * kv2 + M[5][2] * kv3 + M[5][3] * kv4 + M[5][4] * kv5);
			tmp = V(t) + H * (M[5][0] + M[5][1] + M[5][2] + M[5][3] + M[5][4]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn6 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km6 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh6 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv6 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[6][0] * kn1 + M[6][1] * kn2 + M[6][2] * kn3 + M[6][3] * kn4 + M[6][4] * kn5 + M[6][5] * kn6);
			mmp = V(m) + H * (M[6][0] * km1 + M[6][1] * km2 + M[6][2] * km3 + M[6][3] * km4 + M[6][4] * km5 + M[6][5] * km6);
			hmp = V(h) + H * (M[6][0] * kh1 + M[6][1] * kh2 + M[6][2] * kh3 + M[6][3] * kh4 + M[6][4] * kh5 + M[6][5] * kh6);
			vmp = V(v) + H * (M[6][0] * kv1 + M[6][1] * kv2 + M[6][2] * kv3 + M[6][3] * kv4 + M[6][4] * kv5 + M[6][5] * kv6);
			tmp = V(t) + H * (M[6][0] + M[6][1] + M[6][2] + M[6][3] + M[6][4] + M[6][5]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn7 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km7 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh7 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv7 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[7][0] * kn1 + M[7][1] * kn2 + M[7][2] * kn3 + M[7][3] * kn4 + M[7][4] * kn5 + M[7][5] * kn6 + M[7][6] * kn7);
			mmp = V(m) + H * (M[7][0] * km1 + M[7][1] * km2 + M[7][2] * km3 + M[7][3] * km4 + M[7][4] * km5 + M[7][5] * km6 + M[7][6] * km7);
			hmp = V(h) + H * (M[7][0] * kh1 + M[7][1] * kh2 + M[7][2] * kh3 + M[7][3] * kh4 + M[7][4] * kh5 + M[7][5] * kh6 + M[7][6] * kh7);
			vmp = V(v) + H * (M[7][0] * kv1 + M[7][1] * kv2 + M[7][2] * kv3 + M[7][3] * kv4 + M[7][4] * kv5 + M[7][5] * kv6 + M[7][6] * kv7);
			tmp = V(t) + H * (M[7][0] + M[7][1] + M[7][2] + M[7][3] + M[7][4] + M[7][5] + M[7][6]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn8 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km8 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh8 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv8 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[8][0] * kn1 + M[8][1] * kn2 + M[8][2] * kn3 + M[8][3] * kn4 + M[8][4] * kn5 + M[8][5] * kn6 + M[8][6] * kn7 + M[8][7] * kn8);
			mmp = V(m) + H * (M[8][0] * km1 + M[8][1] * km2 + M[8][2] * km3 + M[8][3] * km4 + M[8][4] * km5 + M[8][5] * km6 + M[8][6] * km7 + M[8][7] * km8);
			hmp = V(h) + H * (M[8][0] * kh1 + M[8][1] * kh2 + M[8][2] * kh3 + M[8][3] * kh4 + M[8][4] * kh5 + M[8][5] * kh6 + M[8][6] * kh7 + M[8][7] * kh8);
			vmp = V(v) + H * (M[8][0] * kv1 + M[8][1] * kv2 + M[8][2] * kv3 + M[8][3] * kv4 + M[8][4] * kv5 + M[8][5] * kv6 + M[8][6] * kv7 + M[8][7] * kv8);
			tmp = V(t) + H * (M[8][0] + M[8][1] + M[8][2] + M[8][3] + M[8][4] + M[8][5] + M[8][6] + M[8][7]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn9 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km9 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh9 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kv9 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[9][0] * kn1 + M[9][1] * kn2 + M[9][2] * kn3 + M[9][3] * kn4 + M[9][4] * kn5 + M[9][5] * kn6 + M[9][6] * kn7 + M[9][7] * kn8 + M[9][8] * kn9);
			mmp = V(m) + H * (M[9][0] * km1 + M[9][1] * km2 + M[9][2] * km3 + M[9][3] * km4 + M[9][4] * km5 + M[9][5] * km6 + M[9][6] * km7 + M[9][7] * km8 + M[9][8] * km9);
			hmp = V(h) + H * (M[9][0] * kh1 + M[9][1] * kh2 + M[9][2] * kh3 + M[9][3] * kh4 + M[9][4] * kh5 + M[9][5] * kh6 + M[9][6] * kh7 + M[9][7] * kh8 + M[9][8] * kh9);
			vmp = V(v) + H * (M[9][0] * kv1 + M[9][1] * kv2 + M[9][2] * kv3 + M[9][3] * kv4 + M[9][4] * kv5 + M[9][5] * kv6 + M[9][6] * kv7 + M[9][7] * kv8 + M[9][8] * kv9);
			tmp = V(t) + H * (M[9][0] + M[9][1] + M[9][2] + M[9][3] + M[9][4] + M[9][5] + M[9][6] + M[9][7] + M[9][8]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA0 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA0 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA0 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kvA0 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[10][0] * kn1 + M[10][1] * kn2 + M[10][2] * kn3 + M[10][3] * kn4 + M[10][4] * kn5 + M[10][5] * kn6 + M[10][6] * kn7 + M[10][7] * kn8 + M[10][8] * kn9 + M[10][9] * knA0);
			mmp = V(m) + H * (M[10][0] * km1 + M[10][1] * km2 + M[10][2] * km3 + M[10][3] * km4 + M[10][4] * km5 + M[10][5] * km6 + M[10][6] * km7 + M[10][7] * km8 + M[10][8] * km9 + M[10][9] * kmA0);
			hmp = V(h) + H * (M[10][0] * kh1 + M[10][1] * kh2 + M[10][2] * kh3 + M[10][3] * kh4 + M[10][4] * kh5 + M[10][5] * kh6 + M[10][6] * kh7 + M[10][7] * kh8 + M[10][8] * kh9 + M[10][9] * khA0);
			vmp = V(v) + H * (M[10][0] * kv1 + M[10][1] * kv2 + M[10][2] * kv3 + M[10][3] * kv4 + M[10][4] * kv5 + M[10][5] * kv6 + M[10][6] * kv7 + M[10][7] * kv8 + M[10][8] * kv9 + M[10][9] * kvA0);
			tmp = V(t) + H * (M[10][0] + M[10][1] + M[10][2] + M[10][3] + M[10][4] + M[10][5] + M[10][6] + M[10][7] + M[10][8] + M[10][9]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA1 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA1 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA1 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kvA1 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[11][0] * kn1 + M[11][1] * kn2 + M[11][2] * kn3 + M[11][3] * kn4 + M[11][4] * kn5 + M[11][5] * kn6 + M[11][6] * kn7 + M[11][7] * kn8 + M[11][8] * kn9 + M[11][9] * knA0 + M[11][10] * knA1);
			mmp = V(m) + H * (M[11][0] * km1 + M[11][1] * km2 + M[11][2] * km3 + M[11][3] * km4 + M[11][4] * km5 + M[11][5] * km6 + M[11][6] * km7 + M[11][7] * km8 + M[11][8] * km9 + M[11][9] * kmA0 + M[11][10] * kmA1);
			hmp = V(h) + H * (M[11][0] * kh1 + M[11][1] * kh2 + M[11][2] * kh3 + M[11][3] * kh4 + M[11][4] * kh5 + M[11][5] * kh6 + M[11][6] * kh7 + M[11][7] * kh8 + M[11][8] * kh9 + M[11][9] * khA0 + M[11][10] * khA1);
			vmp = V(v) + H * (M[11][0] * kv1 + M[11][1] * kv2 + M[11][2] * kv3 + M[11][3] * kv4 + M[11][4] * kv5 + M[11][5] * kv6 + M[11][6] * kv7 + M[11][7] * kv8 + M[11][8] * kv9 + M[11][9] * kvA0 + M[11][10] * kvA1);
			tmp = V(t) + H * (M[11][0] + M[11][1] + M[11][2] + M[11][3] + M[11][4] + M[11][5] + M[11][6] + M[11][7] + M[11][8] + M[11][9] + M[11][10]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA2 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA2 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA2 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kvA2 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[12][0] * kn1 + M[12][1] * kn2 + M[12][2] * kn3 + M[12][3] * kn4 + M[12][4] * kn5 + M[12][5] * kn6 + M[12][6] * kn7 + M[12][7] * kn8 + M[12][8] * kn9 + M[12][9] * knA0 + M[12][10] * knA1 + M[12][11] * knA2);
			mmp = V(m) + H * (M[12][0] * km1 + M[12][1] * km2 + M[12][2] * km3 + M[12][3] * km4 + M[12][4] * km5 + M[12][5] * km6 + M[12][6] * km7 + M[12][7] * km8 + M[12][8] * km9 + M[12][9] * kmA0 + M[12][10] * kmA1 + M[12][11] * kmA2);
			hmp = V(h) + H * (M[12][0] * kh1 + M[12][1] * kh2 + M[12][2] * kh3 + M[12][3] * kh4 + M[12][4] * kh5 + M[12][5] * kh6 + M[12][6] * kh7 + M[12][7] * kh8 + M[12][8] * kh9 + M[12][9] * khA0 + M[12][10] * khA1 + M[12][11] * khA2);
			vmp = V(v) + H * (M[12][0] * kv1 + M[12][1] * kv2 + M[12][2] * kv3 + M[12][3] * kv4 + M[12][4] * kv5 + M[12][5] * kv6 + M[12][6] * kv7 + M[12][7] * kv8 + M[12][8] * kv9 + M[12][9] * kvA0 + M[12][10] * kvA1 + M[12][11] * kvA2);
			tmp = V(t) + H * (M[12][0] + M[12][1] + M[12][2] + M[12][3] + M[12][4] + M[12][5] + M[12][6] + M[12][7] + M[12][8] + M[12][9] + M[12][10] + M[12][11]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA3 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA3 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA3 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			Vnext(i) = P(Idc) + (fmodf((tmp - P(Idel)) > 0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), 1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : 0.0f);

			numb kvA3 = (Vnext(i) - I_K - I_Na - I_L) / P(C);


			Vnext(n) = V(n) + H * (B[0][0] * kn1 + B[0][1] * kn2 + B[0][2] * kn3 + B[0][3] * kn4 + B[0][4] * kn5 + B[0][5] * kn6 + B[0][6] * kn7 + B[0][7] * kn8 + B[0][8] * kn9 + B[0][9] * knA0 + B[0][10] * knA1 + B[0][11] * knA2 + B[0][12] * knA3);
			Vnext(m) = V(m) + H * (B[0][0] * km1 + B[0][1] * km2 + B[0][2] * km3 + B[0][3] * km4 + B[0][4] * km5 + B[0][5] * km6 + B[0][6] * km7 + B[0][7] * km8 + B[0][8] * km9 + B[0][9] * kmA0 + B[0][10] * kmA1 + B[0][11] * kmA2 + B[0][12] * kmA3);
			Vnext(h) = V(h) + H * (B[0][0] * kh1 + B[0][1] * kh2 + B[0][2] * kh3 + B[0][3] * kh4 + B[0][4] * kh5 + B[0][5] * kh6 + B[0][6] * kh7 + B[0][7] * kh8 + B[0][8] * kh9 + B[0][9] * khA0 + B[0][10] * khA1 + B[0][11] * khA2 + B[0][12] * kh3);
			Vnext(v) = V(v) + H * (B[0][0] * kv1 + B[0][1] * kv2 + B[0][2] * kv3 + B[0][3] * kv4 + B[0][4] * kv5 + B[0][5] * kv6 + B[0][6] * kv7 + B[0][7] * kv8 + B[0][8] * kv9 + B[0][9] * kvA0 + B[0][10] * kvA1 + B[0][11] * kvA2 + B[0][12] * kvA3);
			Vnext(t) = V(t) + H * (B[0][0] + B[0][1] + B[0][2] + B[0][3] + B[0][4] + B[0][5] + B[0][6] + B[0][7] + B[0][8] + B[0][9] + B[0][10] + B[0][11] + B[0][12]);
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

		ifMETHOD(P(method), ExplicitDormandPrince8)
		{
			numb M[13][12] = { {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0.05555555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0.02083333333333, 0.0625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0.03125, 0, 0.09375, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0.3125, 0, -1.171875, 1.171875, 0, 0, 0, 0, 0, 0, 0, 0},
								{0.0375, 0, 0, 0.1875, 0.15, 0, 0, 0, 0, 0, 0, 0},
								{0.04791013711111, 0, 0, 0.1122487127778, -0.02550567377778, 0.01284682388889, 0, 0, 0, 0, 0, 0},
								{0.01691798978729, 0, 0, 0.387848278486, 0.0359773698515, 0.1969702142157, -0.1727138523405, 0, 0, 0, 0, 0},
								{0.06909575335919, 0, 0, -0.6342479767289, -0.1611975752246, 0.1386503094588, 0.9409286140358, 0.2116363264819, 0, 0, 0, 0},
								{0.183556996839, 0, 0, -2.468768084316, -0.2912868878163, -0.02647302023312, 2.847838764193, 0.2813873314699, 0.1237448998633, 0, 0, 0},
								{-1.215424817396, 0, 0, 16.67260866595, 0.9157418284168, -6.056605804357, -16.00357359416, 14.8493030863, -13.37157573529, 5.13418264818, 0, 0},
								{0.2588609164383, 0, 0, -4.774485785489, -0.435093013777, -3.049483332072, 5.577920039936, 6.155831589861, -5.062104586737, 2.193926173181, 0.1346279986593, 0},
								{0.8224275996265, 0, 0, -11.65867325728, -0.7576221166909, 0.7139735881596, 12.07577498689, -2.12765911392, 1.990166207049, -0.234286471544, 0.1758985777079, 0} };

			numb B[2][14] = { {0.04174749114153, 0, 0, 0, 0, -0.05545232861124, 0.2393128072012, 0.7035106694034, -0.7597596138145, 0.6605630309223, 0.1581874825101, -0.2381095387529, 0.25, 0},
								{0.02955321367635, 0, 0, 0, 0, -0.8286062764878, 0.3112409000511, 2.4673451906, -2.546941651842, 1.443548583677, 0.07941559588113, 0.04444444444444, 0, 0} };

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

			numb imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (V(t) - P(Idel)));

			numb kv1 = (imp - I_K - I_Na - I_L) / P(C);

			numb nmp = V(n) + H * M[1][0] * kn1;
			numb mmp = V(m) + H * M[1][0] * km1;
			numb hmp = V(h) + H * M[1][0] * kh1;
			numb vmp = V(v) + H * M[1][0] * kv1;
			numb tmp = V(t) + H * M[1][0];

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

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kv2 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[2][0] * kn1 + M[2][1] * kn2);
			mmp = V(m) + H * (M[2][0] * km1 + M[2][1] * km2);
			hmp = V(h) + H * (M[2][0] * kh1 + M[2][1] * kh2);
			vmp = V(v) + H * (M[2][0] * kv1 + M[2][1] * kv2);
			tmp = V(t) + H * (M[2][0] + M[2][1]);


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

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kv3 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[3][0] * kn1 + M[3][1] * kn2 + M[3][2] * kn3);
			mmp = V(m) + H * (M[3][0] * km1 + M[2][1] * km2 + M[3][2] * km3);
			hmp = V(h) + H * (M[3][0] * kh1 + M[2][1] * kh2 + M[3][2] * kh3);
			vmp = V(v) + H * (M[3][0] * kv1 + M[2][1] * kv2 + M[3][2] * kv3);
			tmp = V(t) + H * (M[3][0] + M[3][1] + M[3][2]);

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

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kv4 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[4][0] * kn1 + M[4][1] * kn2 + M[4][2] * kn3 + M[4][3] * kn4);
			mmp = V(m) + H * (M[4][0] * km1 + M[4][1] * km2 + M[4][2] * km3 + M[4][3] * km4);
			hmp = V(h) + H * (M[4][0] * kh1 + M[4][1] * kh2 + M[4][2] * kh3 + M[4][3] * kh4);
			vmp = V(v) + H * (M[4][0] * kv1 + M[4][1] * kv2 + M[4][2] * kv3 + M[4][3] * kv4);
			tmp = V(t) + H * (M[4][0] + M[4][1] + M[4][2] + M[4][3]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn5 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km5 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh5 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kv5 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[5][0] * kn1 + M[5][1] * kn2 + M[5][2] * kn3 + M[5][3] * kn4 + M[5][4] * kn5);
			mmp = V(m) + H * (M[5][0] * km1 + M[5][1] * km2 + M[5][2] * km3 + M[5][3] * km4 + M[5][4] * km5);
			hmp = V(h) + H * (M[5][0] * kh1 + M[5][1] * kh2 + M[5][2] * kh3 + M[5][3] * kh4 + M[5][4] * kh5);
			vmp = V(v) + H * (M[5][0] * kv1 + M[5][1] * kv2 + M[5][2] * kv3 + M[5][3] * kv4 + M[5][4] * kv5);
			tmp = V(t) + H * (M[5][0] + M[5][1] + M[5][2] + M[5][3] + M[5][4]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn6 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km6 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh6 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kv6 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[6][0] * kn1 + M[6][1] * kn2 + M[6][2] * kn3 + M[6][3] * kn4 + M[6][4] * kn5 + M[6][5] * kn6);
			mmp = V(m) + H * (M[6][0] * km1 + M[6][1] * km2 + M[6][2] * km3 + M[6][3] * km4 + M[6][4] * km5 + M[6][5] * km6);
			hmp = V(h) + H * (M[6][0] * kh1 + M[6][1] * kh2 + M[6][2] * kh3 + M[6][3] * kh4 + M[6][4] * kh5 + M[6][5] * kh6);
			vmp = V(v) + H * (M[6][0] * kv1 + M[6][1] * kv2 + M[6][2] * kv3 + M[6][3] * kv4 + M[6][4] * kv5 + M[6][5] * kv6);
			tmp = V(t) + H * (M[6][0] + M[6][1] + M[6][2] + M[6][3] + M[6][4] + M[6][5]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn7 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km7 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh7 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kv7 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[7][0] * kn1 + M[7][1] * kn2 + M[7][2] * kn3 + M[7][3] * kn4 + M[7][4] * kn5 + M[7][5] * kn6 + M[7][6] * kn7);
			mmp = V(m) + H * (M[7][0] * km1 + M[7][1] * km2 + M[7][2] * km3 + M[7][3] * km4 + M[7][4] * km5 + M[7][5] * km6 + M[7][6] * km7);
			hmp = V(h) + H * (M[7][0] * kh1 + M[7][1] * kh2 + M[7][2] * kh3 + M[7][3] * kh4 + M[7][4] * kh5 + M[7][5] * kh6 + M[7][6] * kh7);
			vmp = V(v) + H * (M[7][0] * kv1 + M[7][1] * kv2 + M[7][2] * kv3 + M[7][3] * kv4 + M[7][4] * kv5 + M[7][5] * kv6 + M[7][6] * kv7);
			tmp = V(t) + H * (M[7][0] + M[7][1] + M[7][2] + M[7][3] + M[7][4] + M[7][5] + M[7][6]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn8 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km8 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh8 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kv8 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[8][0] * kn1 + M[8][1] * kn2 + M[8][2] * kn3 + M[8][3] * kn4 + M[8][4] * kn5 + M[8][5] * kn6 + M[8][6] * kn7 + M[8][7] * kn8);
			mmp = V(m) + H * (M[8][0] * km1 + M[8][1] * km2 + M[8][2] * km3 + M[8][3] * km4 + M[8][4] * km5 + M[8][5] * km6 + M[8][6] * km7 + M[8][7] * km8);
			hmp = V(h) + H * (M[8][0] * kh1 + M[8][1] * kh2 + M[8][2] * kh3 + M[8][3] * kh4 + M[8][4] * kh5 + M[8][5] * kh6 + M[8][6] * kh7 + M[8][7] * kh8);
			vmp = V(v) + H * (M[8][0] * kv1 + M[8][1] * kv2 + M[8][2] * kv3 + M[8][3] * kv4 + M[8][4] * kv5 + M[8][5] * kv6 + M[8][6] * kv7 + M[8][7] * kv8);
			tmp = V(t) + H * (M[8][0] + M[8][1] + M[8][2] + M[8][3] + M[8][4] + M[8][5] + M[8][6] + M[8][7]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn9 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km9 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh9 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kv9 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[9][0] * kn1 + M[9][1] * kn2 + M[9][2] * kn3 + M[9][3] * kn4 + M[9][4] * kn5 + M[9][5] * kn6 + M[9][6] * kn7 + M[9][7] * kn8 + M[9][8] * kn9);
			mmp = V(m) + H * (M[9][0] * km1 + M[9][1] * km2 + M[9][2] * km3 + M[9][3] * km4 + M[9][4] * km5 + M[9][5] * km6 + M[9][6] * km7 + M[9][7] * km8 + M[9][8] * km9);
			hmp = V(h) + H * (M[9][0] * kh1 + M[9][1] * kh2 + M[9][2] * kh3 + M[9][3] * kh4 + M[9][4] * kh5 + M[9][5] * kh6 + M[9][6] * kh7 + M[9][7] * kh8 + M[9][8] * kh9);
			vmp = V(v) + H * (M[9][0] * kv1 + M[9][1] * kv2 + M[9][2] * kv3 + M[9][3] * kv4 + M[9][4] * kv5 + M[9][5] * kv6 + M[9][6] * kv7 + M[9][7] * kv8 + M[9][8] * kv9);
			tmp = V(t) + H * (M[9][0] + M[9][1] + M[9][2] + M[9][3] + M[9][4] + M[9][5] + M[9][6] + M[9][7] + M[9][8]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA0 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA0 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA0 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kvA0 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[10][0] * kn1 + M[10][1] * kn2 + M[10][2] * kn3 + M[10][3] * kn4 + M[10][4] * kn5 + M[10][5] * kn6 + M[10][6] * kn7 + M[10][7] * kn8 + M[10][8] * kn9 + M[10][9] * knA0);
			mmp = V(m) + H * (M[10][0] * km1 + M[10][1] * km2 + M[10][2] * km3 + M[10][3] * km4 + M[10][4] * km5 + M[10][5] * km6 + M[10][6] * km7 + M[10][7] * km8 + M[10][8] * km9 + M[10][9] * kmA0);
			hmp = V(h) + H * (M[10][0] * kh1 + M[10][1] * kh2 + M[10][2] * kh3 + M[10][3] * kh4 + M[10][4] * kh5 + M[10][5] * kh6 + M[10][6] * kh7 + M[10][7] * kh8 + M[10][8] * kh9 + M[10][9] * khA0);
			vmp = V(v) + H * (M[10][0] * kv1 + M[10][1] * kv2 + M[10][2] * kv3 + M[10][3] * kv4 + M[10][4] * kv5 + M[10][5] * kv6 + M[10][6] * kv7 + M[10][7] * kv8 + M[10][8] * kv9 + M[10][9] * kvA0);
			tmp = V(t) + H * (M[10][0] + M[10][1] + M[10][2] + M[10][3] + M[10][4] + M[10][5] + M[10][6] + M[10][7] + M[10][8] + M[10][9]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA1 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA1 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA1 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kvA1 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[11][0] * kn1 + M[11][1] * kn2 + M[11][2] * kn3 + M[11][3] * kn4 + M[11][4] * kn5 + M[11][5] * kn6 + M[11][6] * kn7 + M[11][7] * kn8 + M[11][8] * kn9 + M[11][9] * knA0 + M[11][10] * knA1);
			mmp = V(m) + H * (M[11][0] * km1 + M[11][1] * km2 + M[11][2] * km3 + M[11][3] * km4 + M[11][4] * km5 + M[11][5] * km6 + M[11][6] * km7 + M[11][7] * km8 + M[11][8] * km9 + M[11][9] * kmA0 + M[11][10] * kmA1);
			hmp = V(h) + H * (M[11][0] * kh1 + M[11][1] * kh2 + M[11][2] * kh3 + M[11][3] * kh4 + M[11][4] * kh5 + M[11][5] * kh6 + M[11][6] * kh7 + M[11][7] * kh8 + M[11][8] * kh9 + M[11][9] * khA0 + M[11][10] * khA1);
			vmp = V(v) + H * (M[11][0] * kv1 + M[11][1] * kv2 + M[11][2] * kv3 + M[11][3] * kv4 + M[11][4] * kv5 + M[11][5] * kv6 + M[11][6] * kv7 + M[11][7] * kv8 + M[11][8] * kv9 + M[11][9] * kvA0 + M[11][10] * kvA1);
			tmp = V(t) + H * (M[11][0] + M[11][1] + M[11][2] + M[11][3] + M[11][4] + M[11][5] + M[11][6] + M[11][7] + M[11][8] + M[11][9] + M[11][10]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA2 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA2 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA2 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kvA2 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[12][0] * kn1 + M[12][1] * kn2 + M[12][2] * kn3 + M[12][3] * kn4 + M[12][4] * kn5 + M[12][5] * kn6 + M[12][6] * kn7 + M[12][7] * kn8 + M[12][8] * kn9 + M[12][9] * knA0 + M[12][10] * knA1 + M[12][11] * knA2);
			mmp = V(m) + H * (M[12][0] * km1 + M[12][1] * km2 + M[12][2] * km3 + M[12][3] * km4 + M[12][4] * km5 + M[12][5] * km6 + M[12][6] * km7 + M[12][7] * km8 + M[12][8] * km9 + M[12][9] * kmA0 + M[12][10] * kmA1 + M[12][11] * kmA2);
			hmp = V(h) + H * (M[12][0] * kh1 + M[12][1] * kh2 + M[12][2] * kh3 + M[12][3] * kh4 + M[12][4] * kh5 + M[12][5] * kh6 + M[12][6] * kh7 + M[12][7] * kh8 + M[12][8] * kh9 + M[12][9] * khA0 + M[12][10] * khA1 + M[12][11] * khA2);
			vmp = V(v) + H * (M[12][0] * kv1 + M[12][1] * kv2 + M[12][2] * kv3 + M[12][3] * kv4 + M[12][4] * kv5 + M[12][5] * kv6 + M[12][6] * kv7 + M[12][7] * kv8 + M[12][8] * kv9 + M[12][9] * kvA0 + M[12][10] * kvA1 + M[12][11] * kvA2);
			tmp = V(t) + H * (M[12][0] + M[12][1] + M[12][2] + M[12][3] + M[12][4] + M[12][5] + M[12][6] + M[12][7] + M[12][8] + M[12][9] + M[12][10] + M[12][11]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA3 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA3 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA3 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			Vnext(i) = P(Idc) + P(Iamp) * sinf(2.0f * 3.141592f * P(Ifreq) * (tmp - P(Idel)));

			numb kvA3 = (Vnext(i) - I_K - I_Na - I_L) / P(C);


			Vnext(n) = V(n) + H * (B[0][0] * kn1 + B[0][1] * kn2 + B[0][2] * kn3 + B[0][3] * kn4 + B[0][4] * kn5 + B[0][5] * kn6 + B[0][6] * kn7 + B[0][7] * kn8 + B[0][8] * kn9 + B[0][9] * knA0 + B[0][10] * knA1 + B[0][11] * knA2 + B[0][12] * knA3);
			Vnext(m) = V(m) + H * (B[0][0] * km1 + B[0][1] * km2 + B[0][2] * km3 + B[0][3] * km4 + B[0][4] * km5 + B[0][5] * km6 + B[0][6] * km7 + B[0][7] * km8 + B[0][8] * km9 + B[0][9] * kmA0 + B[0][10] * kmA1 + B[0][11] * kmA2 + B[0][12] * kmA3);
			Vnext(h) = V(h) + H * (B[0][0] * kh1 + B[0][1] * kh2 + B[0][2] * kh3 + B[0][3] * kh4 + B[0][4] * kh5 + B[0][5] * kh6 + B[0][6] * kh7 + B[0][7] * kh8 + B[0][8] * kh9 + B[0][9] * khA0 + B[0][10] * khA1 + B[0][11] * khA2 + B[0][12] * kh3);
			Vnext(v) = V(v) + H * (B[0][0] * kv1 + B[0][1] * kv2 + B[0][2] * kv3 + B[0][3] * kv4 + B[0][4] * kv5 + B[0][5] * kv6 + B[0][6] * kv7 + B[0][7] * kv8 + B[0][8] * kv9 + B[0][9] * kvA0 + B[0][10] * kvA1 + B[0][11] * kvA2 + B[0][12] * kvA3);
			Vnext(t) = V(t) + H * (B[0][0] + B[0][1] + B[0][2] + B[0][3] + B[0][4] + B[0][5] + B[0][6] + B[0][7] + B[0][8] + B[0][9] + B[0][10] + B[0][11] + B[0][12]);
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

		ifMETHOD(P(method), ExplicitDormandPrince8)
		{
			numb M[13][12] = { {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0.05555555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0.02083333333333, 0.0625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0.03125, 0, 0.09375, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0.3125, 0, -1.171875, 1.171875, 0, 0, 0, 0, 0, 0, 0, 0},
								{0.0375, 0, 0, 0.1875, 0.15, 0, 0, 0, 0, 0, 0, 0},
								{0.04791013711111, 0, 0, 0.1122487127778, -0.02550567377778, 0.01284682388889, 0, 0, 0, 0, 0, 0},
								{0.01691798978729, 0, 0, 0.387848278486, 0.0359773698515, 0.1969702142157, -0.1727138523405, 0, 0, 0, 0, 0},
								{0.06909575335919, 0, 0, -0.6342479767289, -0.1611975752246, 0.1386503094588, 0.9409286140358, 0.2116363264819, 0, 0, 0, 0},
								{0.183556996839, 0, 0, -2.468768084316, -0.2912868878163, -0.02647302023312, 2.847838764193, 0.2813873314699, 0.1237448998633, 0, 0, 0},
								{-1.215424817396, 0, 0, 16.67260866595, 0.9157418284168, -6.056605804357, -16.00357359416, 14.8493030863, -13.37157573529, 5.13418264818, 0, 0},
								{0.2588609164383, 0, 0, -4.774485785489, -0.435093013777, -3.049483332072, 5.577920039936, 6.155831589861, -5.062104586737, 2.193926173181, 0.1346279986593, 0},
								{0.8224275996265, 0, 0, -11.65867325728, -0.7576221166909, 0.7139735881596, 12.07577498689, -2.12765911392, 1.990166207049, -0.234286471544, 0.1758985777079, 0} };

			numb B[2][14] = { {0.04174749114153, 0, 0, 0, 0, -0.05545232861124, 0.2393128072012, 0.7035106694034, -0.7597596138145, 0.6605630309223, 0.1581874825101, -0.2381095387529, 0.25, 0},
								{0.02955321367635, 0, 0, 0, 0, -0.8286062764878, 0.3112409000511, 2.4673451906, -2.546941651842, 1.443548583677, 0.07941559588113, 0.04444444444444, 0, 0} };

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

			numb imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (V(t) - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (V(t) - P(Idel)) + 1.0f) / 2.0f)));

			numb kv1 = (imp - I_K - I_Na - I_L) / P(C);

			numb nmp = V(n) + H * M[1][0] * kn1;
			numb mmp = V(m) + H * M[1][0] * km1;
			numb hmp = V(h) + H * M[1][0] * kh1;
			numb vmp = V(v) + H * M[1][0] * kv1;
			numb tmp = V(t) + H * M[1][0];

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

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kv2 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[2][0] * kn1 + M[2][1] * kn2);
			mmp = V(m) + H * (M[2][0] * km1 + M[2][1] * km2);
			hmp = V(h) + H * (M[2][0] * kh1 + M[2][1] * kh2);
			vmp = V(v) + H * (M[2][0] * kv1 + M[2][1] * kv2);
			tmp = V(t) + H * (M[2][0] + M[2][1]);


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

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kv3 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[3][0] * kn1 + M[3][1] * kn2 + M[3][2] * kn3);
			mmp = V(m) + H * (M[3][0] * km1 + M[2][1] * km2 + M[3][2] * km3);
			hmp = V(h) + H * (M[3][0] * kh1 + M[2][1] * kh2 + M[3][2] * kh3);
			vmp = V(v) + H * (M[3][0] * kv1 + M[2][1] * kv2 + M[3][2] * kv3);
			tmp = V(t) + H * (M[3][0] + M[3][1] + M[3][2]);

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

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kv4 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[4][0] * kn1 + M[4][1] * kn2 + M[4][2] * kn3 + M[4][3] * kn4);
			mmp = V(m) + H * (M[4][0] * km1 + M[4][1] * km2 + M[4][2] * km3 + M[4][3] * km4);
			hmp = V(h) + H * (M[4][0] * kh1 + M[4][1] * kh2 + M[4][2] * kh3 + M[4][3] * kh4);
			vmp = V(v) + H * (M[4][0] * kv1 + M[4][1] * kv2 + M[4][2] * kv3 + M[4][3] * kv4);
			tmp = V(t) + H * (M[4][0] + M[4][1] + M[4][2] + M[4][3]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn5 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km5 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh5 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kv5 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[5][0] * kn1 + M[5][1] * kn2 + M[5][2] * kn3 + M[5][3] * kn4 + M[5][4] * kn5);
			mmp = V(m) + H * (M[5][0] * km1 + M[5][1] * km2 + M[5][2] * km3 + M[5][3] * km4 + M[5][4] * km5);
			hmp = V(h) + H * (M[5][0] * kh1 + M[5][1] * kh2 + M[5][2] * kh3 + M[5][3] * kh4 + M[5][4] * kh5);
			vmp = V(v) + H * (M[5][0] * kv1 + M[5][1] * kv2 + M[5][2] * kv3 + M[5][3] * kv4 + M[5][4] * kv5);
			tmp = V(t) + H * (M[5][0] + M[5][1] + M[5][2] + M[5][3] + M[5][4]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn6 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km6 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh6 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kv6 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[6][0] * kn1 + M[6][1] * kn2 + M[6][2] * kn3 + M[6][3] * kn4 + M[6][4] * kn5 + M[6][5] * kn6);
			mmp = V(m) + H * (M[6][0] * km1 + M[6][1] * km2 + M[6][2] * km3 + M[6][3] * km4 + M[6][4] * km5 + M[6][5] * km6);
			hmp = V(h) + H * (M[6][0] * kh1 + M[6][1] * kh2 + M[6][2] * kh3 + M[6][3] * kh4 + M[6][4] * kh5 + M[6][5] * kh6);
			vmp = V(v) + H * (M[6][0] * kv1 + M[6][1] * kv2 + M[6][2] * kv3 + M[6][3] * kv4 + M[6][4] * kv5 + M[6][5] * kv6);
			tmp = V(t) + H * (M[6][0] + M[6][1] + M[6][2] + M[6][3] + M[6][4] + M[6][5]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn7 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km7 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh7 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kv7 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[7][0] * kn1 + M[7][1] * kn2 + M[7][2] * kn3 + M[7][3] * kn4 + M[7][4] * kn5 + M[7][5] * kn6 + M[7][6] * kn7);
			mmp = V(m) + H * (M[7][0] * km1 + M[7][1] * km2 + M[7][2] * km3 + M[7][3] * km4 + M[7][4] * km5 + M[7][5] * km6 + M[7][6] * km7);
			hmp = V(h) + H * (M[7][0] * kh1 + M[7][1] * kh2 + M[7][2] * kh3 + M[7][3] * kh4 + M[7][4] * kh5 + M[7][5] * kh6 + M[7][6] * kh7);
			vmp = V(v) + H * (M[7][0] * kv1 + M[7][1] * kv2 + M[7][2] * kv3 + M[7][3] * kv4 + M[7][4] * kv5 + M[7][5] * kv6 + M[7][6] * kv7);
			tmp = V(t) + H * (M[7][0] + M[7][1] + M[7][2] + M[7][3] + M[7][4] + M[7][5] + M[7][6]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn8 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km8 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh8 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kv8 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[8][0] * kn1 + M[8][1] * kn2 + M[8][2] * kn3 + M[8][3] * kn4 + M[8][4] * kn5 + M[8][5] * kn6 + M[8][6] * kn7 + M[8][7] * kn8);
			mmp = V(m) + H * (M[8][0] * km1 + M[8][1] * km2 + M[8][2] * km3 + M[8][3] * km4 + M[8][4] * km5 + M[8][5] * km6 + M[8][6] * km7 + M[8][7] * km8);
			hmp = V(h) + H * (M[8][0] * kh1 + M[8][1] * kh2 + M[8][2] * kh3 + M[8][3] * kh4 + M[8][4] * kh5 + M[8][5] * kh6 + M[8][6] * kh7 + M[8][7] * kh8);
			vmp = V(v) + H * (M[8][0] * kv1 + M[8][1] * kv2 + M[8][2] * kv3 + M[8][3] * kv4 + M[8][4] * kv5 + M[8][5] * kv6 + M[8][6] * kv7 + M[8][7] * kv8);
			tmp = V(t) + H * (M[8][0] + M[8][1] + M[8][2] + M[8][3] + M[8][4] + M[8][5] + M[8][6] + M[8][7]);

			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb kn9 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb km9 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb kh9 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kv9 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[9][0] * kn1 + M[9][1] * kn2 + M[9][2] * kn3 + M[9][3] * kn4 + M[9][4] * kn5 + M[9][5] * kn6 + M[9][6] * kn7 + M[9][7] * kn8 + M[9][8] * kn9);
			mmp = V(m) + H * (M[9][0] * km1 + M[9][1] * km2 + M[9][2] * km3 + M[9][3] * km4 + M[9][4] * km5 + M[9][5] * km6 + M[9][6] * km7 + M[9][7] * km8 + M[9][8] * km9);
			hmp = V(h) + H * (M[9][0] * kh1 + M[9][1] * kh2 + M[9][2] * kh3 + M[9][3] * kh4 + M[9][4] * kh5 + M[9][5] * kh6 + M[9][6] * kh7 + M[9][7] * kh8 + M[9][8] * kh9);
			vmp = V(v) + H * (M[9][0] * kv1 + M[9][1] * kv2 + M[9][2] * kv3 + M[9][3] * kv4 + M[9][4] * kv5 + M[9][5] * kv6 + M[9][6] * kv7 + M[9][7] * kv8 + M[9][8] * kv9);
			tmp = V(t) + H * (M[9][0] + M[9][1] + M[9][2] + M[9][3] + M[9][4] + M[9][5] + M[9][6] + M[9][7] + M[9][8]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA0 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA0 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA0 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kvA0 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[10][0] * kn1 + M[10][1] * kn2 + M[10][2] * kn3 + M[10][3] * kn4 + M[10][4] * kn5 + M[10][5] * kn6 + M[10][6] * kn7 + M[10][7] * kn8 + M[10][8] * kn9 + M[10][9] * knA0);
			mmp = V(m) + H * (M[10][0] * km1 + M[10][1] * km2 + M[10][2] * km3 + M[10][3] * km4 + M[10][4] * km5 + M[10][5] * km6 + M[10][6] * km7 + M[10][7] * km8 + M[10][8] * km9 + M[10][9] * kmA0);
			hmp = V(h) + H * (M[10][0] * kh1 + M[10][1] * kh2 + M[10][2] * kh3 + M[10][3] * kh4 + M[10][4] * kh5 + M[10][5] * kh6 + M[10][6] * kh7 + M[10][7] * kh8 + M[10][8] * kh9 + M[10][9] * khA0);
			vmp = V(v) + H * (M[10][0] * kv1 + M[10][1] * kv2 + M[10][2] * kv3 + M[10][3] * kv4 + M[10][4] * kv5 + M[10][5] * kv6 + M[10][6] * kv7 + M[10][7] * kv8 + M[10][8] * kv9 + M[10][9] * kvA0);
			tmp = V(t) + H * (M[10][0] + M[10][1] + M[10][2] + M[10][3] + M[10][4] + M[10][5] + M[10][6] + M[10][7] + M[10][8] + M[10][9]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA1 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA1 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA1 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kvA1 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[11][0] * kn1 + M[11][1] * kn2 + M[11][2] * kn3 + M[11][3] * kn4 + M[11][4] * kn5 + M[11][5] * kn6 + M[11][6] * kn7 + M[11][7] * kn8 + M[11][8] * kn9 + M[11][9] * knA0 + M[11][10] * knA1);
			mmp = V(m) + H * (M[11][0] * km1 + M[11][1] * km2 + M[11][2] * km3 + M[11][3] * km4 + M[11][4] * km5 + M[11][5] * km6 + M[11][6] * km7 + M[11][7] * km8 + M[11][8] * km9 + M[11][9] * kmA0 + M[11][10] * kmA1);
			hmp = V(h) + H * (M[11][0] * kh1 + M[11][1] * kh2 + M[11][2] * kh3 + M[11][3] * kh4 + M[11][4] * kh5 + M[11][5] * kh6 + M[11][6] * kh7 + M[11][7] * kh8 + M[11][8] * kh9 + M[11][9] * khA0 + M[11][10] * khA1);
			vmp = V(v) + H * (M[11][0] * kv1 + M[11][1] * kv2 + M[11][2] * kv3 + M[11][3] * kv4 + M[11][4] * kv5 + M[11][5] * kv6 + M[11][6] * kv7 + M[11][7] * kv8 + M[11][8] * kv9 + M[11][9] * kvA0 + M[11][10] * kvA1);
			tmp = V(t) + H * (M[11][0] + M[11][1] + M[11][2] + M[11][3] + M[11][4] + M[11][5] + M[11][6] + M[11][7] + M[11][8] + M[11][9] + M[11][10]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA2 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA2 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA2 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			imp = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kvA2 = (imp - I_K - I_Na - I_L) / P(C);

			nmp = V(n) + H * (M[12][0] * kn1 + M[12][1] * kn2 + M[12][2] * kn3 + M[12][3] * kn4 + M[12][4] * kn5 + M[12][5] * kn6 + M[12][6] * kn7 + M[12][7] * kn8 + M[12][8] * kn9 + M[12][9] * knA0 + M[12][10] * knA1 + M[12][11] * knA2);
			mmp = V(m) + H * (M[12][0] * km1 + M[12][1] * km2 + M[12][2] * km3 + M[12][3] * km4 + M[12][4] * km5 + M[12][5] * km6 + M[12][6] * km7 + M[12][7] * km8 + M[12][8] * km9 + M[12][9] * kmA0 + M[12][10] * kmA1 + M[12][11] * kmA2);
			hmp = V(h) + H * (M[12][0] * kh1 + M[12][1] * kh2 + M[12][2] * kh3 + M[12][3] * kh4 + M[12][4] * kh5 + M[12][5] * kh6 + M[12][6] * kh7 + M[12][7] * kh8 + M[12][8] * kh9 + M[12][9] * khA0 + M[12][10] * khA1 + M[12][11] * khA2);
			vmp = V(v) + H * (M[12][0] * kv1 + M[12][1] * kv2 + M[12][2] * kv3 + M[12][3] * kv4 + M[12][4] * kv5 + M[12][5] * kv6 + M[12][6] * kv7 + M[12][7] * kv8 + M[12][8] * kv9 + M[12][9] * kvA0 + M[12][10] * kvA1 + M[12][11] * kvA2);
			tmp = V(t) + H * (M[12][0] + M[12][1] + M[12][2] + M[12][3] + M[12][4] + M[12][5] + M[12][6] + M[12][7] + M[12][8] + M[12][9] + M[12][10] + M[12][11]);


			alpha_n = 0.01 * ((10 - vmp) / (exp((10 - vmp) / 10) - 1));
			beta_n = 0.125 * exp(-vmp / 80);
			alpha_m = 0.1 * ((25 - vmp) / (exp((25 - vmp) / 10) - 1));
			beta_m = 4 * exp(-vmp / 18);
			alpha_h = 0.07 * exp(-vmp / 20);
			beta_h = 1 / (exp((30 - vmp) / 10) + 1);

			numb knA3 = alpha_n * (1 - nmp) - beta_n * nmp;
			numb kmA3 = alpha_m * (1 - mmp) - beta_m * mmp;
			numb khA3 = alpha_h * (1 - hmp) - beta_h * hmp;

			I_Na = (mmp * mmp * mmp) * P(G_Na) * hmp * (vmp - P(E_Na));
			I_K = (nmp * nmp * nmp * nmp) * P(G_K) * (vmp - P(E_K));
			I_L = P(G_leak) * (vmp - P(E_leak));

			Vnext(i) = P(Idc) + P(Iamp) * ((4.0f * P(Ifreq) * (tmp - P(Idel)) - 2.0f * floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Ifreq) * (tmp - P(Idel)) + 1.0f) / 2.0f)));

			numb kvA3 = (Vnext(i) - I_K - I_Na - I_L) / P(C);


			Vnext(n) = V(n) + H * (B[0][0] * kn1 + B[0][1] * kn2 + B[0][2] * kn3 + B[0][3] * kn4 + B[0][4] * kn5 + B[0][5] * kn6 + B[0][6] * kn7 + B[0][7] * kn8 + B[0][8] * kn9 + B[0][9] * knA0 + B[0][10] * knA1 + B[0][11] * knA2 + B[0][12] * knA3);
			Vnext(m) = V(m) + H * (B[0][0] * km1 + B[0][1] * km2 + B[0][2] * km3 + B[0][3] * km4 + B[0][4] * km5 + B[0][5] * km6 + B[0][6] * km7 + B[0][7] * km8 + B[0][8] * km9 + B[0][9] * kmA0 + B[0][10] * kmA1 + B[0][11] * kmA2 + B[0][12] * kmA3);
			Vnext(h) = V(h) + H * (B[0][0] * kh1 + B[0][1] * kh2 + B[0][2] * kh3 + B[0][3] * kh4 + B[0][4] * kh5 + B[0][5] * kh6 + B[0][6] * kh7 + B[0][7] * kh8 + B[0][8] * kh9 + B[0][9] * khA0 + B[0][10] * khA1 + B[0][11] * khA2 + B[0][12] * kh3);
			Vnext(v) = V(v) + H * (B[0][0] * kv1 + B[0][1] * kv2 + B[0][2] * kv3 + B[0][3] * kv4 + B[0][4] * kv5 + B[0][5] * kv6 + B[0][6] * kv7 + B[0][7] * kv8 + B[0][8] * kv9 + B[0][9] * kvA0 + B[0][10] * kvA1 + B[0][11] * kvA2 + B[0][12] * kvA3);
			Vnext(t) = V(t) + H * (B[0][0] + B[0][1] + B[0][2] + B[0][3] + B[0][4] + B[0][5] + B[0][6] + B[0][7] + B[0][8] + B[0][9] + B[0][10] + B[0][11] + B[0][12]);
		}
	}
}

#undef name