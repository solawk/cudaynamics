#include "hodgkin_huxley.h"

#define name hodgkin_huxley

namespace attributes
{
	enum variables { v, n, m, h, i, t };
	enum parameters { G_Na, G_leak, G_K, E_Na, E_leak, E_K, C, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, method, COUNT };
	enum waveforms { square, sine, triangle };
	enum methods { ExplicitEuler, SemiExplicitEuler, ImplicitEuler, ExplicitMidpoint, ImplicitMidpoint, ExplicitRungeKutta4, ExplicitDormandPrince8, VariableSymmetryCD };
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
	const numb h = H;
	const int N = 5;
	const numb p[13] = { P(C), P(G_Na), P(E_Na), P(G_K), P(E_K), P(G_leak), P(E_leak), P(Idc), P(Iamp), P(Idel), P(Idf), P(Ifreq), P(symmetry) };
	numb v[N] = { V(v), V(n), V(m), V(h), V(t) };

	ifSIGNAL(P(signal), square)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			numb I = p[7] + (fmod((v[4] - p[9]) > (numb)0.0 ? (v[4] - p[9]) : (p[10] / p[11] + p[9] - v[4]), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
			numb I_Na = (v[2] * v[2] * v[2]) * p[1] * v[3] * (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			Vnext(v) = v[0] + h * ((I - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - v[0]) / (exp(((numb)10.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-v[0] / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - v[0]) / (exp(((numb)25.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-v[0] / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-v[0] / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - v[0]) / (numb)10.0) + (numb)1.0);

			Vnext(n) = v[1] + h * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			Vnext(m) = v[2] + h * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			Vnext(h) = v[3] + h * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);
			Vnext(t) = v[4] + h;

			Vnext(i) = p[7] + (fmod((Vnext(t) - p[9]) > (numb)0.0 ? (Vnext(t) - p[9]) : (p[10] / p[11] + p[9] - Vnext(t)), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
		}

		ifMETHOD(P(method), SemiExplicitEuler)
		{
			numb I = p[7] + (fmod((v[4] - p[9]) > (numb)0.0 ? (v[4] - p[9]) : (p[10] / p[11] + p[9] - v[4]), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
			numb I_Na = (v[2] * v[2] * v[2]) * p[1] * v[3] * (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			Vnext(v) = v[0] + h * ((I - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - Vnext(v)) / (exp(((numb)10.0 - Vnext(v)) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-Vnext(v) / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - Vnext(v)) / (exp(((numb)25.0 - Vnext(v)) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-Vnext(v) / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-Vnext(v) / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - Vnext(v)) / (numb)10.0) + (numb)1.0);

			Vnext(n) = v[1] + h * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			Vnext(m) = v[2] + h * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			Vnext(h) = v[3] + h * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);
			Vnext(t) = v[4] + h;

			Vnext(i) = p[7] + (fmod((Vnext(t) - p[9]) > (numb)0.0 ? (Vnext(t) - p[9]) : (p[10] / p[11] + p[9] - Vnext(t)), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
		}

		ifMETHOD(P(method), ImplicitEuler)
		{
			numb Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I, I_Na, I_K, I_L;
			numb alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[0][4] = (numb)0.0;

			J[1][2] = (numb)0.0;
			J[1][3] = (numb)0.0;
			J[1][4] = (numb)0.0;

			J[2][1] = (numb)0.0;
			J[2][3] = (numb)0.0;
			J[2][4] = (numb)0.0;

			J[3][1] = (numb)0.0;
			J[3][2] = (numb)0.0;
			J[3][4] = (numb)0.0;

			J[4][0] = (numb)0.0;
			J[4][1] = (numb)0.0;
			J[4][2] = (numb)0.0;
			J[4][3] = (numb)0.0;
			J[4][4] = (numb)0.0;

			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = -(p[1] * Z[2] * Z[2] * Z[2] * Z[3] + p[3] * Z[1] * Z[1] * Z[1] * Z[1] + p[5]) / p[0];

				J[0][1] = -(p[3] * (numb)4.0 * Z[1] * Z[1] * Z[1] * (Z[0] - p[4])) / p[0];
				J[0][2] = -(p[1] * (numb)3.0 * Z[2] * Z[2] * Z[3] * (Z[0] - p[2])) / p[0];
				J[0][3] = -(p[1] * Z[2] * Z[2] * Z[2] * (Z[0] - p[2])) / p[0];

				J[1][0] = (-(numb)0.01 * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)10.0 - Z[0]) * exp(((numb)10.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[1]) - (-(numb)0.0015625 * exp(-Z[0] / (numb)80.0)) * Z[1];
				J[1][1] = -((numb)0.01 * (((numb)10.0 - Z[0]) / (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)0.125 * exp(-Z[0] / (numb)80.0));

				J[2][0] = (-(numb)0.1 * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)25.0 - Z[0]) * exp(((numb)25.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[2]) - (-((numb)4.0 / (numb)18.0) * exp(-Z[0] / (numb)18.0)) * Z[2];
				J[2][2] = -((numb)0.1 * (((numb)25.0 - Z[0]) / (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)4.0 * exp(-Z[0] / (numb)18.0));

				J[3][0] = (-(numb)0.0035 * exp(-Z[0] / (numb)20.0)) * ((numb)1.0 - Z[3]) - ((numb)0.1 * exp(((numb)30.0 - Z[0]) / (numb)10.0) / ((exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0) * (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0))) * Z[3];
				J[3][3] = -((numb)0.07 * exp(-Z[0] / (numb)20.0) + (numb)1.0 / (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0));

				for (i = 0; i < N; i++) {
					for (j = 0; j < N; j++) {
						if (i == j)
							W[i][j] = (numb)1.0 - h * J[i][j];
						else
							W[i][j] = - h * J[i][j];
					}
				}

				I = p[7] + (fmod((Z[4] - p[9]) > (numb)0.0 ? (Z[4] - p[9]) : (p[10] / p[11] + p[9] - Z[4]), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);

				I_Na = (Z[2] * Z[2] * Z[2]) * Z[3] * p[1] * (Z[0] - p[2]);
				I_K = (Z[1] * Z[1] * Z[1] * Z[1]) * p[3] * (Z[0] - p[4]);
				I_L = p[5] * (Z[0] - p[6]);

				W[0][N] = v[0] - Z[0] + h * ((I - I_Na - I_K - I_L) / p[0]);

				alpha_n = (numb)0.01 * (((numb)10.0 - Z[0]) / (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0));
				beta_n = (numb)0.125 * exp(-Z[0] / (numb)80.0);
				alpha_m = (numb)0.1 * (((numb)25.0 - Z[0]) / (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0));
				beta_m = (numb)4.0 * exp(-Z[0] / (numb)18.0);
				alpha_h = (numb)0.07 * exp(-Z[0] / (numb)20.0);
				beta_h = (numb)1.0 / (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0);

				W[1][N] = v[1] - Z[1] + h * (alpha_n * ((numb)1.0 - Z[1]) - beta_n * Z[1]);
				W[2][N] = v[2] - Z[2] + h * (alpha_m * ((numb)1.0 - Z[2]) - beta_m * Z[2]);
				W[3][N] = v[3] - Z[3] + h * (alpha_h * ((numb)1.0 - Z[3]) - beta_h * Z[3]);
				W[4][N] = v[4] - Z[4] + h * ((numb)1.0);

				int k; 
				numb b, d;

				for (k = 0; k <= N - 2; k++) {

					int l = k;

					for (i = k + 1; i <= N - 1; i++) {
						if (fabs(W[i][k]) > fabs(W[l][k])) {
							l = i;
						}
					}
					if (l != k) {
						for (j = 0; j <= N; j++) {
							if ((j == 0) || (j >= k)) {
								b = W[k][j];
								W[k][j] = W[l][j];
								W[l][j] = b;
							}
						}
					}

					d = (numb)1.0 / W[k][k];
					for (i = (k + 1); i <= (N - 1); i++) {
						if (W[i][k] == (numb)0.0) {
							continue;
						}
						b = W[i][k] * d;
						for (j = k; j <= N; j++) {
							if (W[k][j] != (numb)0.0) {
								W[i][j] = W[i][j] - b * W[k][j];
							}
						}
					}
				}

				for (i = N; i >= 2; i--) {
					for (j = 1; j <= i - 1; j++) {
						b = W[i - j - 1][i - 1] / W[i - 1][i - 1];
						W[i - j - 1][N] = W[i - j - 1][N] - b * W[i - 1][N];
					}
					W[i - 1][N] = W[i - 1][N] / W[i - 1][i - 1];
				}
				W[0][N] = W[0][N] / W[0][0];

				for (i = 0; i < N; i++)
					Zn[i] = Z[i] + W[i][N];

				dz = (numb)0.0;
				for (i = 0; i < N; i++) {
					numb diff = Zn[i] - Z[i];
					dz += diff * diff;
				}
				dz = sqrt(dz);

				for (i = 0; i < N; i++)
					Z[i] = Zn[i];

				nnewt++;
			}

			Vnext(v) = Zn[0];
			Vnext(n) = Zn[1];
			Vnext(m) = Zn[2];
			Vnext(h) = Zn[3];
			Vnext(t) = Zn[4];

			Vnext(i) = p[7] + (fmod((Vnext(t) - p[9]) > (numb)0.0 ? (Vnext(t) - p[9]) : (p[10] / p[11] + p[9] - Vnext(t)), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
		}

		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = p[7] + (fmod((v[4] - p[9]) > (numb)0.0 ? (v[4] - p[9]) : (p[10] / p[11] + p[9] - v[4]), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
			numb I_Na = (v[2] * v[2] * v[2]) * p[1] * v[3] * (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			numb vmp = v[0] + h * (numb)0.5 * ((imp - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - v[0]) / (exp(((numb)10.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-v[0] / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - v[0]) / (exp(((numb)25.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-v[0] / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-v[0] / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - v[0]) / (numb)10.0) + (numb)1.0);

			numb nmp = v[1] + h * (numb)0.5 * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			numb mmp = v[2] + h * (numb)0.5 * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			numb hmp = v[3] + h * (numb)0.5 * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);
			numb tmp = v[4] + h * (numb)0.5;

			numb I = p[7] + (fmod((tmp - p[9]) > (numb)0.0 ? (tmp - p[9]) : (p[10] / p[11] + p[9] - tmp), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
			I_Na = (mmp * mmp * mmp) * p[1] * hmp * (vmp - p[2]);
			I_K = (nmp * nmp * nmp * nmp) * p[3] * (vmp - p[4]);
			I_L = p[5] * (vmp - p[6]);

			Vnext(v) = v[0] + h * ((I - I_K - I_Na - I_L) / p[0]);

			alpha_n = (numb)0.01 * (((numb)10.0 - vmp) / (exp(((numb)10.0 - vmp) / (numb)10.0) - (numb)1.0));
			beta_n = (numb)0.125 * exp(-vmp / (numb)80.0);
			alpha_m = (numb)0.1 * (((numb)25.0 - vmp) / (exp(((numb)25.0 - vmp) / (numb)10.0) - (numb)1.0));
			beta_m = (numb)4.0 * exp(-vmp / (numb)18.0);
			alpha_h = (numb)0.07 * exp(-vmp / (numb)20.0);
			beta_h = (numb)1.0 / (exp(((numb)30.0 - vmp) / (numb)10.0) + (numb)1.0);

			Vnext(n) = v[1] + h * (alpha_n * ((numb)1.0 - nmp) - beta_n * nmp);
			Vnext(m) = v[2] + h * (alpha_m * ((numb)1.0 - mmp) - beta_m * mmp);
			Vnext(h) = v[3] + h * (alpha_h * ((numb)1.0 - hmp) - beta_h * hmp);
			Vnext(t) = v[4] + h;

			Vnext(i) = p[7] + (fmod((Vnext(t) - p[9]) > (numb)0.0 ? (Vnext(t) - p[9]) : (p[10] / p[11] + p[9] - Vnext(t)), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
		}

		ifMETHOD(P(method), ImplicitMidpoint)
		{
			numb a[N], Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I, I_Na, I_K, I_L;
			numb alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[0][4] = (numb)0.0;

			J[1][2] = (numb)0.0;
			J[1][3] = (numb)0.0;
			J[1][4] = (numb)0.0;

			J[2][1] = (numb)0.0;
			J[2][3] = (numb)0.0;
			J[2][4] = (numb)0.0;

			J[3][1] = (numb)0.0;
			J[3][2] = (numb)0.0;
			J[3][4] = (numb)0.0;

			J[4][0] = (numb)0.0;
			J[4][1] = (numb)0.0;
			J[4][2] = (numb)0.0;
			J[4][3] = (numb)0.0;
			J[4][4] = (numb)0.0;

			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = -(p[1] * Z[2] * Z[2] * Z[2] * Z[3] + p[3] * Z[1] * Z[1] * Z[1] * Z[1] + p[5]) / p[0];

				J[0][1] = -(p[3] * (numb)4.0 * Z[1] * Z[1] * Z[1] * (Z[0] - p[4])) / p[0];
				J[0][2] = -(p[1] * (numb)3.0 * Z[2] * Z[2] * Z[3] * (Z[0] - p[2])) / p[0];
				J[0][3] = -(p[1] * Z[2] * Z[2] * Z[2] * (Z[0] - p[2])) / p[0];

				J[1][0] = (-(numb)0.01 * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)10.0 - Z[0]) * exp(((numb)10.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[1]) - (-(numb)0.0015625 * exp(-Z[0] / (numb)80.0)) * Z[1];
				J[1][1] = -((numb)0.01 * (((numb)10.0 - Z[0]) / (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)0.125 * exp(-Z[0] / (numb)80.0));

				J[2][0] = (-(numb)0.1 * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)25.0 - Z[0]) * exp(((numb)25.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[2]) - (-((numb)4.0 / (numb)18.0) * exp(-Z[0] / (numb)18.0)) * Z[2];
				J[2][2] = -((numb)0.1 * (((numb)25.0 - Z[0]) / (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)4.0 * exp(-Z[0] / (numb)18.0));

				J[3][0] = (-(numb)0.0035 * exp(-Z[0] / (numb)20.0)) * ((numb)1.0 - Z[3]) - ((numb)0.1 * exp(((numb)30.0 - Z[0]) / (numb)10.0) / ((exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0) * (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0))) * Z[3];
				J[3][3] = -((numb)0.07 * exp(-Z[0] / (numb)20.0) + (numb)1.0 / (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0));

				for (i = 0; i < N; i++) {
					for (j = 0; j < N; j++) {
						if (i == j)
							W[i][j] = (numb)1.0 - (numb)0.5 * h * J[i][j];
						else
							W[i][j] = -(numb)0.5 * h * J[i][j];
					}
				}

				for (i = 0; i < N; i++)
					a[i] = (numb)0.5 * (v[i] + Z[i]);

				I = p[7] + (fmod((a[4] - p[9]) > (numb)0.0 ? (a[4] - p[9]) : (p[10] / p[11] + p[9] - a[4]), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);

				I_Na = (a[2] * a[2] * a[2]) * a[3] * p[1] * (a[0] - p[2]);
				I_K = (a[1] * a[1] * a[1] * a[1]) * p[3] * (a[0] - p[4]);
				I_L = p[5] * (a[0] - p[6]);

				W[0][5] = v[0] - Z[0] + h * ((I - I_Na - I_K - I_L) / p[0]);

				alpha_n = (numb)0.01 * (((numb)10.0 - a[0]) / (exp(((numb)10.0 - a[0]) / (numb)10.0) - (numb)1.0));
				beta_n = (numb)0.125 * exp(-a[0] / (numb)80.0);
				alpha_m = (numb)0.1 * (((numb)25.0 - a[0]) / (exp(((numb)25.0 - a[0]) / (numb)10.0) - (numb)1.0));
				beta_m = (numb)4.0 * exp(-a[0] / (numb)18.0);
				alpha_h = (numb)0.07 * exp(-a[0] / (numb)20.0);
				beta_h = (numb)1.0 / (exp(((numb)30.0 - a[0]) / (numb)10.0) + (numb)1.0);

				W[1][N] = v[1] - Z[1] + h * (alpha_n * ((numb)1.0 - a[1]) - beta_n * a[1]);
				W[2][N] = v[2] - Z[2] + h * (alpha_m * ((numb)1.0 - a[2]) - beta_m * a[2]);
				W[3][N] = v[3] - Z[3] + h * (alpha_h * ((numb)1.0 - a[3]) - beta_h * a[3]);
				W[4][N] = v[4] - Z[4] + h * ((numb)1.0);

				int k; 
				numb b, d;

				for (k = 0; k <= N - 2; k++) {

					int l = k;

					for (i = k + 1; i <= N - 1; i++) {
						if (fabs(W[i][k]) > fabs(W[l][k])) {
							l = i;
						}
					}
					if (l != k) {
						for (j = 0; j <= N; j++) {
							if ((j == 0) || (j >= k)) {
								b = W[k][j];
								W[k][j] = W[l][j];
								W[l][j] = b;
							}
						}
					}

					d = (numb)1.0 / W[k][k];
					for (i = (k + 1); i <= (N - 1); i++) {
						if (W[i][k] == (numb)0.0) {
							continue;
						}
						b = W[i][k] * d;
						for (j = k; j <= N; j++) {
							if (W[k][j] != (numb)0.0) {
								W[i][j] = W[i][j] - b * W[k][j];
							}
						}
					}
				}

				for (i = N; i >= 2; i--) {
					for (j = 1; j <= i - 1; j++) {
						b = W[i - j - 1][i - 1] / W[i - 1][i - 1];
						W[i - j - 1][N] = W[i - j - 1][N] - b * W[i - 1][N];
					}
					W[i - 1][N] = W[i - 1][N] / W[i - 1][i - 1];
				}
				W[0][N] = W[0][N] / W[0][0];

				for (i = 0; i < N; i++)
					Zn[i] = Z[i] + W[i][N];

				dz = (numb)0.0;
				for (i = 0; i < N; i++) {
					numb diff = Zn[i] - Z[i];
					dz += diff * diff;
				}
				dz = sqrt(dz);

				for (i = 0; i < N; i++)
					Z[i] = Zn[i];

				nnewt++;
			}

			Vnext(v) = Zn[0];
			Vnext(n) = Zn[1];
			Vnext(m) = Zn[2];
			Vnext(h) = Zn[3];
			Vnext(t) = Zn[4];

			Vnext(i) = p[7] + (fmod((Vnext(t) - p[9]) > (numb)0.0 ? (Vnext(t) - p[9]) : (p[10] / p[11] + p[9] - Vnext(t)), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
		}

		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb k[N][4];
			numb a[N];
			numb I;
			int i, j;

			for (i = 0; i < N; i++)
				a[i] = v[i];

			for (j = 0; j < 4; j++) {

				I = p[7] + (fmod((a[4] - p[9]) > (numb)0.0 ? (a[4] - p[9]) : (p[10] / p[11] + p[9] - a[4]), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);

				numb alpha_n = (numb)0.01 * (((numb)10.0 - a[0]) / (exp(((numb)10.0 - a[0]) / (numb)10.0) - (numb)1.0));
				numb beta_n = (numb)0.125 * exp(-a[0] / (numb)80.0);

				numb alpha_m = (numb)0.1 * (((numb)25.0 - a[0]) / (exp(((numb)25.0 - a[0]) / (numb)10.0) - (numb)1.0));
				numb beta_m = (numb)4.0 * exp(-a[0] / (numb)18.0);

				numb alpha_h = (numb)0.07 * exp(-a[0] / (numb)20.0);
				numb beta_h = (numb)1.0 / (exp(((numb)30.0 - a[0]) / (numb)10.0) + (numb)1.0);

				numb I_Na = a[2] * a[2] * a[2] * a[3] * p[1] * (a[0] - p[2]);
				numb I_K = a[1] * a[1] * a[1] * a[1] * p[3] * (a[0] - p[4]);
				numb I_L = p[5] * (a[0] - p[6]);

				k[0][j] = (I - I_Na - I_K - I_L) / p[0];
				k[1][j] = alpha_n * ((numb)1.0 - a[1]) - beta_n * a[1];
				k[2][j] = alpha_m * ((numb)1.0 - a[2]) - beta_m * a[2];
				k[3][j] = alpha_h * ((numb)1.0 - a[3]) - beta_h * a[3];
				k[4][j] = (numb)1.0;

				if (j == 3) {

					Vnext(v) = v[0] + h * (k[0][0] + (numb)2.0 * k[0][1] + (numb)2.0 * k[0][2] + k[0][3]) / (numb)6.0;
					Vnext(n) = v[1] + h * (k[1][0] + (numb)2.0 * k[1][1] + (numb)2.0 * k[1][2] + k[1][3]) / (numb)6.0;
					Vnext(m) = v[2] + h * (k[2][0] + (numb)2.0 * k[2][1] + (numb)2.0 * k[2][2] + k[2][3]) / (numb)6.0;
					Vnext(h) = v[3] + h * (k[3][0] + (numb)2.0 * k[3][1] + (numb)2.0 * k[3][2] + k[3][3]) / (numb)6.0;
					Vnext(t) = v[4] + h * (k[4][0] + (numb)2.0 * k[4][1] + (numb)2.0 * k[4][2] + k[4][3]) / (numb)6.0;
				}
				else if (j == 2) {

					for (i = 0; i < N; i++)
						a[i] = v[i] + h * k[i][j];
				}
				else {

					for (i = 0; i < N; i++)
						a[i] = v[i] + (numb)0.5 * h * k[i][j];
				}
			}

			Vnext(i) = p[7] + (fmod((Vnext(t) - p[9]) > (numb)0.0 ? (Vnext(t) - p[9]) : (p[10] / p[11] + p[9] - Vnext(t)), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
		}

		ifMETHOD(P(method), ExplicitDormandPrince8)
		{
			const numb M[13][12] = { {(numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.05555555555556, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.02083333333333, (numb)0.0625, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.03125, (numb)0.0, (numb)0.09375, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.3125, (numb)0.0, -(numb)1.171875, (numb)1.171875, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.0375, (numb)0.0, (numb)0.0, (numb)0.1875, (numb)0.15, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.04791013711111, (numb)0.0, (numb)0.0, (numb)0.1122487127778, -(numb)0.02550567377778, (numb)0.01284682388889, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.01691798978729, (numb)0.0, (numb)0.0, (numb)0.387848278486, (numb)0.0359773698515, (numb)0.1969702142157, -(numb)0.1727138523405, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.06909575335919, (numb)0.0, (numb)0.0, -(numb)0.6342479767289, -(numb)0.1611975752246, (numb)0.1386503094588, (numb)0.9409286140358, (numb)0.2116363264819, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.183556996839, (numb)0.0, (numb)0.0, -(numb)2.468768084316, -(numb)0.2912868878163, -(numb)0.02647302023312, (numb)2.847838764193, (numb)0.2813873314699, (numb)0.1237448998633, (numb)0.0, (numb)0.0, (numb)0.0},
								{-(numb)1.215424817396, (numb)0.0, (numb)0.0, (numb)16.67260866595, (numb)0.9157418284168, -(numb)6.056605804357, -(numb)16.00357359416, (numb)14.8493030863, -(numb)13.37157573529, (numb)5.13418264818, (numb)0.0, (numb)0.0},
								{(numb)0.2588609164383, (numb)0.0, (numb)0.0, -(numb)4.774485785489, -(numb)0.435093013777, -(numb)3.049483332072, (numb)5.577920039936, (numb)6.155831589861, -(numb)5.062104586737, (numb)2.193926173181, (numb)0.1346279986593, (numb)0.0},
								{(numb)0.8224275996265, (numb)0.0, (numb)0.0, -(numb)11.65867325728, -(numb)0.7576221166909, (numb)0.7139735881596, (numb)12.07577498689, -(numb)2.12765911392, (numb)1.990166207049, -(numb)0.234286471544, (numb)0.1758985777079, (numb)0.0} };

			const numb b[13] = { (numb)0.04174749114153, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, -(numb)0.05545232861124, (numb)0.2393128072012, (numb)0.7035106694034, -(numb)0.7597596138145, (numb)0.6605630309223, (numb)0.1581874825101, -(numb)0.2381095387529, (numb)0.25 };
			numb y[N], X1[N], X2[N];
			numb k[N][13];
			int i, j, l;
			numb I;

			for (i = 0; i < N; i++)
				X1[i] = v[i];

			for (i = 0; i < 13; i++) {
				I = p[7] + (fmod((X1[4] - p[9]) > (numb)0.0 ? (X1[4] - p[9]) : (p[10] / p[11] + p[9] - X1[4]), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);

				numb alpha_n = (numb)0.01 * (((numb)10.0 - X1[0]) / (exp(((numb)10.0 - X1[0]) / (numb)10.0) - (numb)1.0));
				numb beta_n = (numb)0.125 * exp(-X1[0] / (numb)80.0);

				numb alpha_m = (numb)0.1 * (((numb)25.0 - X1[0]) / (exp(((numb)25.0 - X1[0]) / (numb)10.0) - (numb)1.0));
				numb beta_m = (numb)4.0 * exp(-X1[0] / (numb)18.0);

				numb alpha_h = (numb)0.07 * exp(-X1[0] / (numb)20.0);
				numb beta_h = (numb)1.0 / (exp(((numb)30.0 - X1[0]) / (numb)10.0) + (numb)1.0);

				numb I_Na = X1[2] * X1[2] * X1[2] * X1[3] * p[1] * (X1[0] - p[2]);
				numb I_K = X1[1] * X1[1] * X1[1] * X1[1] * p[3] * (X1[0] - p[4]);
				numb I_L = p[5] * (X1[0] - p[6]);

				k[0][j] = (I - I_Na - I_K - I_L) / p[0];
				k[1][j] = alpha_n * ((numb)1.0 - X1[1]) - beta_n * X1[1];
				k[2][j] = alpha_m * ((numb)1.0 - X1[2]) - beta_m * X1[2];
				k[3][j] = alpha_h * ((numb)1.0 - X1[3]) - beta_h * X1[3];
				k[4][j] = (numb)1.0;

				for (l = 0; l < N; l++)
					X2[l] = 0;

				for (j = 0; j < i + 1; j++)
					for (l = 0; l < N; l++)
						X2[l] += M[i + 1][j] * k[l][j];

				for (l = 0; l < N; l++)
					X1[l] = v[l] + H * X2[l];
			}

			for (l = 0; l < N; l++)
				X2[l] = 0;

			for (i = 0; i < 13; i++)
				for (l = 0; l < N; l++)
					X2[l] += b[i] * k[l][i];

			for (l = 0; l < N; l++)
				y[l] = v[l] + H * X2[l];

			Vnext(v) = y[0];
			Vnext(n) = y[1];
			Vnext(m) = y[2];
			Vnext(h) = y[3];
			Vnext(t) = y[4];

			Vnext(i) = p[7] + (fmod((Vnext(t) - p[9]) > (numb)0.0 ? (Vnext(t) - p[9]) : (p[10] / p[11] + p[9] - Vnext(t)), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
		}

		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = (numb)0.5 * h - p[12];
			numb h2 = (numb)0.5 * h + p[12];

			numb I = p[7] + (fmod((v[4] - p[9]) > (numb)0.0 ? (v[4] - p[9]) : (p[10] / p[11] + p[9] - v[4]), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
			numb I_Na = (v[2] * v[2] * v[2])* p[1]* v[3]* (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			numb vmp = v[0] + h1 * ((I - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - vmp) / (exp(((numb)10.0 - vmp) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-vmp / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - vmp) / (exp(((numb)25.0 - vmp) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-vmp / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-vmp / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - vmp) / (numb)10.0) + (numb)1.0);

			numb nmp = v[1] + h1 * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			numb mmp = v[2] + h1 * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			numb hmp = v[3] + h1 * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);

			Vnext(t) = V(t) + h;

			Vnext(h) = (hmp + h2 * alpha_h) / ((numb)1.0 + h2 * (alpha_h + beta_h));
			Vnext(m) = (mmp + h2 * alpha_m) / ((numb)1.0 + h2 * (alpha_m + beta_m));
			Vnext(n) = (nmp + h2 * alpha_n) / ((numb)1.0 + h2 * (alpha_n + beta_n));

			Vnext(i) = p[7] + (fmod((Vnext(t) - p[9]) > (numb)0.0 ? (Vnext(t) - p[9]) : (p[10] / p[11] + p[9] - Vnext(t)), (numb)1.0 / p[11]) < p[10] / p[11] ? p[8] : (numb)0.0);
			numb x_Na = (Vnext(m) * Vnext(m) * Vnext(m)) * p[1] * Vnext(h);
			numb x_K = (Vnext(n) * Vnext(n) * Vnext(n) * Vnext(n)) * p[3];

			Vnext(v) = (vmp + h2 * p[0] * (Vnext(i) + x_Na * p[2] + x_K * p[4] + p[5] * p[6]) ) / ( 1 + h2 * p[0] * (x_Na + x_K + p[5]) );
		}
	}

	ifSIGNAL(P(signal), sine)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			numb I = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (v[4] - p[9]));
			numb I_Na = (v[2] * v[2] * v[2]) * p[1] * v[3] * (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			Vnext(v) = v[0] + h * ((I - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - v[0]) / (exp(((numb)10.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-v[0] / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - v[0]) / (exp(((numb)25.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-v[0] / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-v[0] / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - v[0]) / (numb)10.0) + (numb)1.0);

			Vnext(n) = v[1] + h * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			Vnext(m) = v[2] + h * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			Vnext(h) = v[3] + h * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);
			Vnext(t) = v[4] + h;

			Vnext(i) = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (Vnext(t) - p[9]));
		}

		ifMETHOD(P(method), SemiExplicitEuler)
		{
			numb I = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (v[4] - p[9]));
			numb I_Na = (v[2] * v[2] * v[2]) * p[1] * v[3] * (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			Vnext(v) = v[0] + h * ((I - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - Vnext(v)) / (exp(((numb)10.0 - Vnext(v)) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-Vnext(v) / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - Vnext(v)) / (exp(((numb)25.0 - Vnext(v)) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-Vnext(v) / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-Vnext(v) / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - Vnext(v)) / (numb)10.0) + (numb)1.0);

			Vnext(n) = v[1] + h * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			Vnext(m) = v[2] + h * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			Vnext(h) = v[3] + h * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);
			Vnext(t) = v[4] + h;

			Vnext(i) = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (Vnext(t) - p[9]));
		}

		ifMETHOD(P(method), ImplicitEuler)
		{
			numb Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I, I_Na, I_K, I_L;
			numb alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[1][2] = (numb)0.0;
			J[1][3] = (numb)0.0;
			J[1][4] = (numb)0.0;

			J[2][1] = (numb)0.0;
			J[2][3] = (numb)0.0;
			J[2][4] = (numb)0.0;

			J[3][1] = (numb)0.0;
			J[3][2] = (numb)0.0;
			J[3][4] = (numb)0.0;

			J[4][0] = (numb)0.0;
			J[4][1] = (numb)0.0;
			J[4][2] = (numb)0.0;
			J[4][3] = (numb)0.0;
			J[4][4] = (numb)0.0;

			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = -(p[1] * Z[2] * Z[2] * Z[2] * Z[3] + p[3] * Z[1] * Z[1] * Z[1] * Z[1] + p[5]) / p[0];

				J[0][1] = -(p[3] * (numb)4.0 * Z[1] * Z[1] * Z[1] * (Z[0] - p[4])) / p[0];
				J[0][2] = -(p[1] * (numb)3.0 * Z[2] * Z[2] * Z[3] * (Z[0] - p[2])) / p[0];
				J[0][3] = -(p[1] * Z[2] * Z[2] * Z[2] * (Z[0] - p[2])) / p[0];
				J[0][4] = (p[8] * cos((numb)2.0 * (numb)3.141592653589793 * p[11] * (Z[4] - p[9])) * (numb)2.0 * (numb)3.141592653589793 * p[11]) / p[0];

				J[1][0] = (-(numb)0.01 * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)10.0 - Z[0]) * exp(((numb)10.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[1]) - (-(numb)0.0015625 * exp(-Z[0] / (numb)80.0)) * Z[1];
				J[1][1] = -((numb)0.01 * (((numb)10.0 - Z[0]) / (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)0.125 * exp(-Z[0] / (numb)80.0));

				J[2][0] = (-(numb)0.1 * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)25.0 - Z[0]) * exp(((numb)25.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[2]) - (-((numb)4.0 / (numb)18.0) * exp(-Z[0] / (numb)18.0)) * Z[2];
				J[2][2] = -((numb)0.1 * (((numb)25.0 - Z[0]) / (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)4.0 * exp(-Z[0] / (numb)18.0));

				J[3][0] = (-(numb)0.0035 * exp(-Z[0] / (numb)20.0)) * ((numb)1.0 - Z[3]) - ((numb)0.1 * exp(((numb)30.0 - Z[0]) / (numb)10.0) / ((exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0) * (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0))) * Z[3];
				J[3][3] = -((numb)0.07 * exp(-Z[0] / (numb)20.0) + (numb)1.0 / (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0));

				for (i = 0; i < N; i++) {
					for (j = 0; j < N; j++) {
						if (i == j)
							W[i][j] = (numb)1.0 - h * J[i][j];
						else
							W[i][j] = -h * J[i][j];
					}
				}

				I = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (Z[4] - p[9]));

				I_Na = (Z[2] * Z[2] * Z[2]) * Z[3] * p[1] * (Z[0] - p[2]);
				I_K = (Z[1] * Z[1] * Z[1] * Z[1]) * p[3] * (Z[0] - p[4]);
				I_L = p[5] * (Z[0] - p[6]);

				W[0][N] = v[0] - Z[0] + h * ((I - I_Na - I_K - I_L) / p[0]);

				alpha_n = (numb)0.01 * (((numb)10.0 - Z[0]) / (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0));
				beta_n = (numb)0.125 * exp(-Z[0] / (numb)80.0);
				alpha_m = (numb)0.1 * (((numb)25.0 - Z[0]) / (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0));
				beta_m = (numb)4.0 * exp(-Z[0] / (numb)18.0);
				alpha_h = (numb)0.07 * exp(-Z[0] / (numb)20.0);
				beta_h = (numb)1.0 / (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0);

				W[1][N] = v[1] - Z[1] + h * (alpha_n * ((numb)1.0 - Z[1]) - beta_n * Z[1]);
				W[2][N] = v[2] - Z[2] + h * (alpha_m * ((numb)1.0 - Z[2]) - beta_m * Z[2]);
				W[3][N] = v[3] - Z[3] + h * (alpha_h * ((numb)1.0 - Z[3]) - beta_h * Z[3]);
				W[4][N] = v[4] - Z[4] + h * ((numb)1.0);

				int k;
				numb b, d;

				for (k = 0; k <= N - 2; k++) {

					int l = k;

					for (i = k + 1; i <= N - 1; i++) {
						if (fabs(W[i][k]) > fabs(W[l][k])) {
							l = i;
						}
					}
					if (l != k) {
						for (j = 0; j <= N; j++) {
							if ((j == 0) || (j >= k)) {
								b = W[k][j];
								W[k][j] = W[l][j];
								W[l][j] = b;
							}
						}
					}

					d = (numb)1.0 / W[k][k];
					for (i = (k + 1); i <= (N - 1); i++) {
						if (W[i][k] == (numb)0.0) {
							continue;
						}
						b = W[i][k] * d;
						for (j = k; j <= N; j++) {
							if (W[k][j] != (numb)0.0) {
								W[i][j] = W[i][j] - b * W[k][j];
							}
						}
					}
				}

				for (i = N; i >= 2; i--) {
					for (j = 1; j <= i - 1; j++) {
						b = W[i - j - 1][i - 1] / W[i - 1][i - 1];
						W[i - j - 1][N] = W[i - j - 1][N] - b * W[i - 1][N];
					}
					W[i - 1][N] = W[i - 1][N] / W[i - 1][i - 1];
				}
				W[0][N] = W[0][N] / W[0][0];

				for (i = 0; i < N; i++)
					Zn[i] = Z[i] + W[i][N];

				dz = (numb)0.0;
				for (i = 0; i < N; i++) {
					numb diff = Zn[i] - Z[i];
					dz += diff * diff;
				}
				dz = sqrt(dz);

				for (i = 0; i < N; i++)
					Z[i] = Zn[i];

				nnewt++;
			}

			Vnext(v) = Zn[0];
			Vnext(n) = Zn[1];
			Vnext(m) = Zn[2];
			Vnext(h) = Zn[3];
			Vnext(t) = Zn[4];

			Vnext(i) = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (Vnext(t) - p[9]));
		}

		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (v[4] - p[9]));
			numb I_Na = (v[2] * v[2] * v[2]) * p[1] * v[3] * (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			numb vmp = v[0] + h * (numb)0.5 * ((imp - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - v[0]) / (exp(((numb)10.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-v[0] / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - v[0]) / (exp(((numb)25.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-v[0] / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-v[0] / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - v[0]) / (numb)10.0) + (numb)1.0);

			numb nmp = v[1] + h * (numb)0.5 * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			numb mmp = v[2] + h * (numb)0.5 * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			numb hmp = v[3] + h * (numb)0.5 * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);
			numb tmp = v[4] + h * (numb)0.5;

			numb I = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (tmp - p[9]));
			I_Na = (mmp * mmp * mmp) * p[1] * hmp * (vmp - p[2]);
			I_K = (nmp * nmp * nmp * nmp) * p[3] * (vmp - p[4]);
			I_L = p[5] * (vmp - p[6]);

			Vnext(v) = v[0] + h * ((I - I_K - I_Na - I_L) / p[0]);

			alpha_n = (numb)0.01 * (((numb)10.0 - vmp) / (exp(((numb)10.0 - vmp) / (numb)10.0) - (numb)1.0));
			beta_n = (numb)0.125 * exp(-vmp / (numb)80.0);
			alpha_m = (numb)0.1 * (((numb)25.0 - vmp) / (exp(((numb)25.0 - vmp) / (numb)10.0) - (numb)1.0));
			beta_m = (numb)4.0 * exp(-vmp / (numb)18.0);
			alpha_h = (numb)0.07 * exp(-vmp / (numb)20.0);
			beta_h = (numb)1.0 / (exp(((numb)30.0 - vmp) / (numb)10.0) + (numb)1.0);

			Vnext(n) = v[1] + h * (alpha_n * ((numb)1.0 - nmp) - beta_n * nmp);
			Vnext(m) = v[2] + h * (alpha_m * ((numb)1.0 - mmp) - beta_m * mmp);
			Vnext(h) = v[3] + h * (alpha_h * ((numb)1.0 - hmp) - beta_h * hmp);
			Vnext(t) = v[4] + h;

			Vnext(i) = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (Vnext(t) - p[9]));
		}

		ifMETHOD(P(method), ImplicitMidpoint)
		{
			numb a[N], Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I, I_Na, I_K, I_L;
			numb alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[1][2] = (numb)0.0;
			J[1][3] = (numb)0.0;
			J[1][4] = (numb)0.0;

			J[2][1] = (numb)0.0;
			J[2][3] = (numb)0.0;
			J[2][4] = (numb)0.0;

			J[3][1] = (numb)0.0;
			J[3][2] = (numb)0.0;
			J[3][4] = (numb)0.0;

			J[4][0] = (numb)0.0;
			J[4][1] = (numb)0.0;
			J[4][2] = (numb)0.0;
			J[4][3] = (numb)0.0;
			J[4][4] = (numb)0.0;

			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = -(p[1] * Z[2] * Z[2] * Z[2] * Z[3] + p[3] * Z[1] * Z[1] * Z[1] * Z[1] + p[5]) / p[0];

				J[0][1] = -(p[3] * (numb)4.0 * Z[1] * Z[1] * Z[1] * (Z[0] - p[4])) / p[0];
				J[0][2] = -(p[1] * (numb)3.0 * Z[2] * Z[2] * Z[3] * (Z[0] - p[2])) / p[0];
				J[0][3] = -(p[1] * Z[2] * Z[2] * Z[2] * (Z[0] - p[2])) / p[0];
				J[0][4] = (p[8] * cos((numb)2.0 * (numb)3.141592653589793 * p[11] * (Z[4] - p[9])) * (numb)2.0 * (numb)3.141592653589793 * p[11]) / p[0];

				J[1][0] = (-(numb)0.01 * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)10.0 - Z[0]) * exp(((numb)10.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[1]) - (-(numb)0.0015625 * exp(-Z[0] / (numb)80.0)) * Z[1];
				J[1][1] = -((numb)0.01 * (((numb)10.0 - Z[0]) / (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)0.125 * exp(-Z[0] / (numb)80.0));

				J[2][0] = (-(numb)0.1 * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)25.0 - Z[0]) * exp(((numb)25.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[2]) - (-((numb)4.0 / (numb)18.0) * exp(-Z[0] / (numb)18.0)) * Z[2];
				J[2][2] = -((numb)0.1 * (((numb)25.0 - Z[0]) / (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)4.0 * exp(-Z[0] / (numb)18.0));

				J[3][0] = (-(numb)0.0035 * exp(-Z[0] / (numb)20.0)) * ((numb)1.0 - Z[3]) - ((numb)0.1 * exp(((numb)30.0 - Z[0]) / (numb)10.0) / ((exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0) * (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0))) * Z[3];
				J[3][3] = -((numb)0.07 * exp(-Z[0] / (numb)20.0) + (numb)1.0 / (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0));

				for (i = 0; i < N; i++) {
					for (j = 0; j < N; j++) {
						if (i == j)
							W[i][j] = (numb)1.0 - (numb)0.5 * h * J[i][j];
						else
							W[i][j] = -(numb)0.5 * h * J[i][j];
					}
				}

				for (i = 0; i < N; i++)
					a[i] = (numb)0.5 * (v[i] + Z[i]);

				I = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (a[4] - p[9]));

				I_Na = (a[2] * a[2] * a[2]) * a[3] * p[1] * (a[0] - p[2]);
				I_K = (a[1] * a[1] * a[1] * a[1]) * p[3] * (a[0] - p[4]);
				I_L = p[5] * (a[0] - p[6]);

				W[0][5] = v[0] - Z[0] + h * ((I - I_Na - I_K - I_L) / p[0]);

				alpha_n = (numb)0.01 * (((numb)10.0 - a[0]) / (exp(((numb)10.0 - a[0]) / (numb)10.0) - (numb)1.0));
				beta_n = (numb)0.125 * exp(-a[0] / (numb)80.0);
				alpha_m = (numb)0.1 * (((numb)25.0 - a[0]) / (exp(((numb)25.0 - a[0]) / (numb)10.0) - (numb)1.0));
				beta_m = (numb)4.0 * exp(-a[0] / (numb)18.0);
				alpha_h = (numb)0.07 * exp(-a[0] / (numb)20.0);
				beta_h = (numb)1.0 / (exp(((numb)30.0 - a[0]) / (numb)10.0) + (numb)1.0);

				W[1][N] = v[1] - Z[1] + h * (alpha_n * ((numb)1.0 - a[1]) - beta_n * a[1]);
				W[2][N] = v[2] - Z[2] + h * (alpha_m * ((numb)1.0 - a[2]) - beta_m * a[2]);
				W[3][N] = v[3] - Z[3] + h * (alpha_h * ((numb)1.0 - a[3]) - beta_h * a[3]);
				W[4][N] = v[4] - Z[4] + h * ((numb)1.0);

				int k;
				numb b, d;

				for (k = 0; k <= N - 2; k++) {

					int l = k;

					for (i = k + 1; i <= N - 1; i++) {
						if (fabs(W[i][k]) > fabs(W[l][k])) {
							l = i;
						}
					}
					if (l != k) {
						for (j = 0; j <= N; j++) {
							if ((j == 0) || (j >= k)) {
								b = W[k][j];
								W[k][j] = W[l][j];
								W[l][j] = b;
							}
						}
					}

					d = (numb)1.0 / W[k][k];
					for (i = (k + 1); i <= (N - 1); i++) {
						if (W[i][k] == (numb)0.0) {
							continue;
						}
						b = W[i][k] * d;
						for (j = k; j <= N; j++) {
							if (W[k][j] != (numb)0.0) {
								W[i][j] = W[i][j] - b * W[k][j];
							}
						}
					}
				}

				for (i = N; i >= 2; i--) {
					for (j = 1; j <= i - 1; j++) {
						b = W[i - j - 1][i - 1] / W[i - 1][i - 1];
						W[i - j - 1][N] = W[i - j - 1][N] - b * W[i - 1][N];
					}
					W[i - 1][N] = W[i - 1][N] / W[i - 1][i - 1];
				}
				W[0][N] = W[0][N] / W[0][0];

				for (i = 0; i < N; i++)
					Zn[i] = Z[i] + W[i][N];

				dz = (numb)0.0;
				for (i = 0; i < N; i++) {
					numb diff = Zn[i] - Z[i];
					dz += diff * diff;
				}
				dz = sqrt(dz);

				for (i = 0; i < N; i++)
					Z[i] = Zn[i];

				nnewt++;
			}

			Vnext(v) = Zn[0];
			Vnext(n) = Zn[1];
			Vnext(m) = Zn[2];
			Vnext(h) = Zn[3];
			Vnext(t) = Zn[4];

			Vnext(i) = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (Vnext(t) - p[9]));
		}

		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb k[N][4];
			numb a[N];
			numb I;
			int i, j;

			for (i = 0; i < N; i++)
				a[i] = v[i];

			for (j = 0; j < 4; j++) {

				I = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (a[4] - p[9]));

				numb alpha_n = (numb)0.01 * (((numb)10.0 - a[0]) / (exp(((numb)10.0 - a[0]) / (numb)10.0) - (numb)1.0));
				numb beta_n = (numb)0.125 * exp(-a[0] / (numb)80.0);

				numb alpha_m = (numb)0.1 * (((numb)25.0 - a[0]) / (exp(((numb)25.0 - a[0]) / (numb)10.0) - (numb)1.0));
				numb beta_m = (numb)4.0 * exp(-a[0] / (numb)18.0);

				numb alpha_h = (numb)0.07 * exp(-a[0] / (numb)20.0);
				numb beta_h = (numb)1.0 / (exp(((numb)30.0 - a[0]) / (numb)10.0) + (numb)1.0);

				numb I_Na = a[2] * a[2] * a[2] * a[3] * p[1] * (a[0] - p[2]);
				numb I_K = a[1] * a[1] * a[1] * a[1] * p[3] * (a[0] - p[4]);
				numb I_L = p[5] * (a[0] - p[6]);

				k[0][j] = (I - I_Na - I_K - I_L) / p[0];
				k[1][j] = alpha_n * ((numb)1.0 - a[1]) - beta_n * a[1];
				k[2][j] = alpha_m * ((numb)1.0 - a[2]) - beta_m * a[2];
				k[3][j] = alpha_h * ((numb)1.0 - a[3]) - beta_h * a[3];
				k[4][j] = (numb)1.0;

				if (j == 3) {

					Vnext(v) = v[0] + h * (k[0][0] + (numb)2.0 * k[0][1] + (numb)2.0 * k[0][2] + k[0][3]) / (numb)6.0;
					Vnext(n) = v[1] + h * (k[1][0] + (numb)2.0 * k[1][1] + (numb)2.0 * k[1][2] + k[1][3]) / (numb)6.0;
					Vnext(m) = v[2] + h * (k[2][0] + (numb)2.0 * k[2][1] + (numb)2.0 * k[2][2] + k[2][3]) / (numb)6.0;
					Vnext(h) = v[3] + h * (k[3][0] + (numb)2.0 * k[3][1] + (numb)2.0 * k[3][2] + k[3][3]) / (numb)6.0;
					Vnext(t) = v[4] + h * (k[4][0] + (numb)2.0 * k[4][1] + (numb)2.0 * k[4][2] + k[4][3]) / (numb)6.0;
				}
				else if (j == 2) {

					for (i = 0; i < N; i++)
						a[i] = v[i] + h * k[i][j];
				}
				else {

					for (i = 0; i < N; i++)
						a[i] = v[i] + (numb)0.5 * h * k[i][j];
				}
			}

			Vnext(i) = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (Vnext(t) - p[9]));
		}

		ifMETHOD(P(method), ExplicitDormandPrince8)
		{
			const numb M[13][12] = { {(numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.05555555555556, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.02083333333333, (numb)0.0625, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.03125, (numb)0.0, (numb)0.09375, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.3125, (numb)0.0, -(numb)1.171875, (numb)1.171875, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.0375, (numb)0.0, (numb)0.0, (numb)0.1875, (numb)0.15, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.04791013711111, (numb)0.0, (numb)0.0, (numb)0.1122487127778, -(numb)0.02550567377778, (numb)0.01284682388889, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.01691798978729, (numb)0.0, (numb)0.0, (numb)0.387848278486, (numb)0.0359773698515, (numb)0.1969702142157, -(numb)0.1727138523405, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.06909575335919, (numb)0.0, (numb)0.0, -(numb)0.6342479767289, -(numb)0.1611975752246, (numb)0.1386503094588, (numb)0.9409286140358, (numb)0.2116363264819, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.183556996839, (numb)0.0, (numb)0.0, -(numb)2.468768084316, -(numb)0.2912868878163, -(numb)0.02647302023312, (numb)2.847838764193, (numb)0.2813873314699, (numb)0.1237448998633, (numb)0.0, (numb)0.0, (numb)0.0},
								{-(numb)1.215424817396, (numb)0.0, (numb)0.0, (numb)16.67260866595, (numb)0.9157418284168, -(numb)6.056605804357, -(numb)16.00357359416, (numb)14.8493030863, -(numb)13.37157573529, (numb)5.13418264818, (numb)0.0, (numb)0.0},
								{(numb)0.2588609164383, (numb)0.0, (numb)0.0, -(numb)4.774485785489, -(numb)0.435093013777, -(numb)3.049483332072, (numb)5.577920039936, (numb)6.155831589861, -(numb)5.062104586737, (numb)2.193926173181, (numb)0.1346279986593, (numb)0.0},
								{(numb)0.8224275996265, (numb)0.0, (numb)0.0, -(numb)11.65867325728, -(numb)0.7576221166909, (numb)0.7139735881596, (numb)12.07577498689, -(numb)2.12765911392, (numb)1.990166207049, -(numb)0.234286471544, (numb)0.1758985777079, (numb)0.0} };

			const numb b[13] = { (numb)0.04174749114153, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, -(numb)0.05545232861124, (numb)0.2393128072012, (numb)0.7035106694034, -(numb)0.7597596138145, (numb)0.6605630309223, (numb)0.1581874825101, -(numb)0.2381095387529, (numb)0.25 };
			numb y[N], X1[N], X2[N];
			numb k[N][13];
			int i, j, l;
			numb I;

			for (i = 0; i < N; i++)
				X1[i] = v[i];

			for (i = 0; i < 13; i++) {

				I = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (X1[4] - p[9]));

				numb alpha_n = (numb)0.01 * (((numb)10.0 - X1[0]) / (exp(((numb)10.0 - X1[0]) / (numb)10.0) - (numb)1.0));
				numb beta_n = (numb)0.125 * exp(-X1[0] / (numb)80.0);

				numb alpha_m = (numb)0.1 * (((numb)25.0 - X1[0]) / (exp(((numb)25.0 - X1[0]) / (numb)10.0) - (numb)1.0));
				numb beta_m = (numb)4.0 * exp(-X1[0] / (numb)18.0);

				numb alpha_h = (numb)0.07 * exp(-X1[0] / (numb)20.0);
				numb beta_h = (numb)1.0 / (exp(((numb)30.0 - X1[0]) / (numb)10.0) + (numb)1.0);

				numb I_Na = X1[2] * X1[2] * X1[2] * X1[3] * p[1] * (X1[0] - p[2]);
				numb I_K = X1[1] * X1[1] * X1[1] * X1[1] * p[3] * (X1[0] - p[4]);
				numb I_L = p[5] * (X1[0] - p[6]);

				k[0][j] = (I - I_Na - I_K - I_L) / p[0];
				k[1][j] = alpha_n * ((numb)1.0 - X1[1]) - beta_n * X1[1];
				k[2][j] = alpha_m * ((numb)1.0 - X1[2]) - beta_m * X1[2];
				k[3][j] = alpha_h * ((numb)1.0 - X1[3]) - beta_h * X1[3];
				k[4][j] = (numb)1.0;

				for (l = 0; l < N; l++)
					X2[l] = 0;

				for (j = 0; j < i + 1; j++)
					for (l = 0; l < N; l++)
						X2[l] += M[i + 1][j] * k[l][j];

				for (l = 0; l < N; l++)
					X1[l] = v[l] + H * X2[l];
			}

			for (l = 0; l < N; l++)
				X2[l] = 0;

			for (i = 0; i < 13; i++)
				for (l = 0; l < N; l++)
					X2[l] += b[i] * k[l][i];

			for (l = 0; l < N; l++)
				y[l] = v[l] + H * X2[l];

			Vnext(v) = y[0];
			Vnext(n) = y[1];
			Vnext(m) = y[2];
			Vnext(h) = y[3];
			Vnext(t) = y[4];

			Vnext(i) = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (Vnext(t) - p[9]));
		}

		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = (numb)0.5 * h - p[12];
			numb h2 = (numb)0.5 * h + p[12];

			numb I = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (v[4] - p[9]));
			numb I_Na = (v[2] * v[2] * v[2]) * p[1] * v[3] * (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			numb vmp = v[0] + h1 * ((I - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - vmp) / (exp(((numb)10.0 - vmp) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-vmp / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - vmp) / (exp(((numb)25.0 - vmp) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-vmp / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-vmp / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - vmp) / (numb)10.0) + (numb)1.0);

			numb nmp = v[1] + h1 * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			numb mmp = v[2] + h1 * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			numb hmp = v[3] + h1 * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);

			Vnext(t) = V(t) + h;

			Vnext(h) = (hmp + h2 * alpha_h) / ((numb)1.0 + h2 * (alpha_h + beta_h));
			Vnext(m) = (mmp + h2 * alpha_m) / ((numb)1.0 + h2 * (alpha_m + beta_m));
			Vnext(n) = (nmp + h2 * alpha_n) / ((numb)1.0 + h2 * (alpha_n + beta_n));

			Vnext(i) = p[7] + p[8] * sin((numb)2.0 * (numb)3.141592653589793 * p[11] * (Vnext(t) - p[9]));
			numb x_Na = (Vnext(m) * Vnext(m) * Vnext(m)) * p[1] * Vnext(h);
			numb x_K = (Vnext(n) * Vnext(n) * Vnext(n) * Vnext(n)) * p[3];

			Vnext(v) = (vmp + h2 * p[0] * (Vnext(i) + x_Na * p[2] + x_K * p[4] + p[5] * p[6])) / (1 + h2 * p[0] * (x_Na + x_K + p[5]));
		}
	}

	ifSIGNAL(P(signal), triangle)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			numb I = p[7] + p[8] * (((numb)4.0 * p[11] * (v[4] - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (v[4] - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (v[4] - p[9]) + (numb)1.0) / (numb)2.0)));
			numb I_Na = (v[2] * v[2] * v[2]) * p[1] * v[3] * (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			Vnext(v) = v[0] + h * ((I - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - v[0]) / (exp(((numb)10.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-v[0] / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - v[0]) / (exp(((numb)25.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-v[0] / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-v[0] / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - v[0]) / (numb)10.0) + (numb)1.0);

			Vnext(n) = v[1] + h * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			Vnext(m) = v[2] + h * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			Vnext(h) = v[3] + h * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);
			Vnext(t) = v[4] + h;

			Vnext(i) = p[7] + p[8] * (((numb)4.0 * p[11] * (Vnext(t) - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)));
		}

		ifMETHOD(P(method), SemiExplicitEuler)
		{
			numb I = p[7] + p[8] * (((numb)4.0 * p[11] * (v[4] - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (v[4] - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (v[4] - p[9]) + (numb)1.0) / (numb)2.0)));
			numb I_Na = (v[2] * v[2] * v[2]) * p[1] * v[3] * (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			Vnext(v) = v[0] + h * ((I - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - Vnext(v)) / (exp(((numb)10.0 - Vnext(v)) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-Vnext(v) / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - Vnext(v)) / (exp(((numb)25.0 - Vnext(v)) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-Vnext(v) / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-Vnext(v) / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - Vnext(v)) / (numb)10.0) + (numb)1.0);

			Vnext(n) = v[1] + h * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			Vnext(m) = v[2] + h * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			Vnext(h) = v[3] + h * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);
			Vnext(t) = v[4] + h;

			Vnext(i) = p[7] + p[8] * (((numb)4.0 * p[11] * (Vnext(t) - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)));
		}

		ifMETHOD(P(method), ImplicitEuler)
		{
			numb Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I, I_Na, I_K, I_L;
			numb alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[1][2] = (numb)0.0;
			J[1][3] = (numb)0.0;
			J[1][4] = (numb)0.0;

			J[2][1] = (numb)0.0;
			J[2][3] = (numb)0.0;
			J[2][4] = (numb)0.0;

			J[3][1] = (numb)0.0;
			J[3][2] = (numb)0.0;
			J[3][4] = (numb)0.0;

			J[4][0] = (numb)0.0;
			J[4][1] = (numb)0.0;
			J[4][2] = (numb)0.0;
			J[4][3] = (numb)0.0;
			J[4][4] = (numb)0.0;

			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = -(p[1] * Z[2] * Z[2] * Z[2] * Z[3] + p[3] * Z[1] * Z[1] * Z[1] * Z[1] + p[5]) / p[0];

				J[0][1] = -(p[3] * (numb)4.0 * Z[1] * Z[1] * Z[1] * (Z[0] - p[4])) / p[0];
				J[0][2] = -(p[1] * (numb)3.0 * Z[2] * Z[2] * Z[3] * (Z[0] - p[2])) / p[0];
				J[0][3] = -(p[1] * Z[2] * Z[2] * Z[2] * (Z[0] - p[2])) / p[0];
				J[0][4] = (p[8] * (numb)4.0 * p[11] * (fmod(floor(((numb)4.0 * p[11] * (Z[4] - p[9]) + (numb)1.0) / (numb)2.0), (numb)2.0) < (numb)1.0 ? (numb)1.0 : -(numb)1.0)) / p[0];

				J[1][0] = (-(numb)0.01 * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)10.0 - Z[0]) * exp(((numb)10.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[1]) - (-(numb)0.0015625 * exp(-Z[0] / (numb)80.0)) * Z[1];
				J[1][1] = -((numb)0.01 * (((numb)10.0 - Z[0]) / (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)0.125 * exp(-Z[0] / (numb)80.0));

				J[2][0] = (-(numb)0.1 * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)25.0 - Z[0]) * exp(((numb)25.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[2]) - (-((numb)4.0 / (numb)18.0) * exp(-Z[0] / (numb)18.0)) * Z[2];
				J[2][2] = -((numb)0.1 * (((numb)25.0 - Z[0]) / (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)4.0 * exp(-Z[0] / (numb)18.0));

				J[3][0] = (-(numb)0.0035 * exp(-Z[0] / (numb)20.0)) * ((numb)1.0 - Z[3]) - ((numb)0.1 * exp(((numb)30.0 - Z[0]) / (numb)10.0) / ((exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0) * (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0))) * Z[3];
				J[3][3] = -((numb)0.07 * exp(-Z[0] / (numb)20.0) + (numb)1.0 / (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0));

				for (i = 0; i < N; i++) {
					for (j = 0; j < N; j++) {
						if (i == j)
							W[i][j] = (numb)1.0 - h * J[i][j];
						else
							W[i][j] = -h * J[i][j];
					}
				}

				I = p[7] + p[8] * (((numb)4.0 * p[11] * (Z[4] - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (Z[4] - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (Z[4] - p[9]) + (numb)1.0) / (numb)2.0)));

				I_Na = (Z[2] * Z[2] * Z[2]) * Z[3] * p[1] * (Z[0] - p[2]);
				I_K = (Z[1] * Z[1] * Z[1] * Z[1]) * p[3] * (Z[0] - p[4]);
				I_L = p[5] * (Z[0] - p[6]);

				W[0][N] = v[0] - Z[0] + h * ((I - I_Na - I_K - I_L) / p[0]);

				alpha_n = (numb)0.01 * (((numb)10.0 - Z[0]) / (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0));
				beta_n = (numb)0.125 * exp(-Z[0] / (numb)80.0);
				alpha_m = (numb)0.1 * (((numb)25.0 - Z[0]) / (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0));
				beta_m = (numb)4.0 * exp(-Z[0] / (numb)18.0);
				alpha_h = (numb)0.07 * exp(-Z[0] / (numb)20.0);
				beta_h = (numb)1.0 / (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0);

				W[1][N] = v[1] - Z[1] + h * (alpha_n * ((numb)1.0 - Z[1]) - beta_n * Z[1]);
				W[2][N] = v[2] - Z[2] + h * (alpha_m * ((numb)1.0 - Z[2]) - beta_m * Z[2]);
				W[3][N] = v[3] - Z[3] + h * (alpha_h * ((numb)1.0 - Z[3]) - beta_h * Z[3]);
				W[4][N] = v[4] - Z[4] + h * ((numb)1.0);

				int k;
				numb b, d;

				for (k = 0; k <= N - 2; k++) {

					int l = k;

					for (i = k + 1; i <= N - 1; i++) {
						if (fabs(W[i][k]) > fabs(W[l][k])) {
							l = i;
						}
					}
					if (l != k) {
						for (j = 0; j <= N; j++) {
							if ((j == 0) || (j >= k)) {
								b = W[k][j];
								W[k][j] = W[l][j];
								W[l][j] = b;
							}
						}
					}

					d = (numb)1.0 / W[k][k];
					for (i = (k + 1); i <= (N - 1); i++) {
						if (W[i][k] == (numb)0.0) {
							continue;
						}
						b = W[i][k] * d;
						for (j = k; j <= N; j++) {
							if (W[k][j] != (numb)0.0) {
								W[i][j] = W[i][j] - b * W[k][j];
							}
						}
					}
				}

				for (i = N; i >= 2; i--) {
					for (j = 1; j <= i - 1; j++) {
						b = W[i - j - 1][i - 1] / W[i - 1][i - 1];
						W[i - j - 1][N] = W[i - j - 1][N] - b * W[i - 1][N];
					}
					W[i - 1][N] = W[i - 1][N] / W[i - 1][i - 1];
				}
				W[0][N] = W[0][N] / W[0][0];

				for (i = 0; i < N; i++)
					Zn[i] = Z[i] + W[i][N];

				dz = (numb)0.0;
				for (i = 0; i < N; i++) {
					numb diff = Zn[i] - Z[i];
					dz += diff * diff;
				}
				dz = sqrt(dz);

				for (i = 0; i < N; i++)
					Z[i] = Zn[i];

				nnewt++;
			}

			Vnext(v) = Zn[0];
			Vnext(n) = Zn[1];
			Vnext(m) = Zn[2];
			Vnext(h) = Zn[3];
			Vnext(t) = Zn[4];

			Vnext(i) = p[7] + p[8] * (((numb)4.0 * p[11] * (Vnext(t) - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)));
		}

		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = p[7] + p[8] * (((numb)4.0 * p[11] * (v[4] - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (v[4] - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (v[4] - p[9]) + (numb)1.0) / (numb)2.0)));
			numb I_Na = (v[2] * v[2] * v[2]) * p[1] * v[3] * (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			numb vmp = v[0] + h * (numb)0.5 * ((imp - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - v[0]) / (exp(((numb)10.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-v[0] / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - v[0]) / (exp(((numb)25.0 - v[0]) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-v[0] / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-v[0] / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - v[0]) / (numb)10.0) + (numb)1.0);

			numb nmp = v[1] + h * (numb)0.5 * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			numb mmp = v[2] + h * (numb)0.5 * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			numb hmp = v[3] + h * (numb)0.5 * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);
			numb tmp = v[4] + h * (numb)0.5;

			numb I = p[7] + p[8] * (((numb)4.0 * p[11] * (tmp - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (tmp - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (tmp - p[9]) + (numb)1.0) / (numb)2.0)));
			I_Na = (mmp * mmp * mmp) * p[1] * hmp * (vmp - p[2]);
			I_K = (nmp * nmp * nmp * nmp) * p[3] * (vmp - p[4]);
			I_L = p[5] * (vmp - p[6]);

			Vnext(v) = v[0] + h * ((I - I_K - I_Na - I_L) / p[0]);

			alpha_n = (numb)0.01 * (((numb)10.0 - vmp) / (exp(((numb)10.0 - vmp) / (numb)10.0) - (numb)1.0));
			beta_n = (numb)0.125 * exp(-vmp / (numb)80.0);
			alpha_m = (numb)0.1 * (((numb)25.0 - vmp) / (exp(((numb)25.0 - vmp) / (numb)10.0) - (numb)1.0));
			beta_m = (numb)4.0 * exp(-vmp / (numb)18.0);
			alpha_h = (numb)0.07 * exp(-vmp / (numb)20.0);
			beta_h = (numb)1.0 / (exp(((numb)30.0 - vmp) / (numb)10.0) + (numb)1.0);

			Vnext(n) = v[1] + h * (alpha_n * ((numb)1.0 - nmp) - beta_n * nmp);
			Vnext(m) = v[2] + h * (alpha_m * ((numb)1.0 - mmp) - beta_m * mmp);
			Vnext(h) = v[3] + h * (alpha_h * ((numb)1.0 - hmp) - beta_h * hmp);
			Vnext(t) = v[4] + h;

			Vnext(i) = p[7] + p[8] * (((numb)4.0 * p[11] * (Vnext(t) - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)));
		}

		ifMETHOD(P(method), ImplicitMidpoint)
		{
			numb a[N], Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I, I_Na, I_K, I_L;
			numb alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[1][2] = (numb)0.0;
			J[1][3] = (numb)0.0;
			J[1][4] = (numb)0.0;

			J[2][1] = (numb)0.0;
			J[2][3] = (numb)0.0;
			J[2][4] = (numb)0.0;

			J[3][1] = (numb)0.0;
			J[3][2] = (numb)0.0;
			J[3][4] = (numb)0.0;

			J[4][0] = (numb)0.0;
			J[4][1] = (numb)0.0;
			J[4][2] = (numb)0.0;
			J[4][3] = (numb)0.0;
			J[4][4] = (numb)0.0;

			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = -(p[1] * Z[2] * Z[2] * Z[2] * Z[3] + p[3] * Z[1] * Z[1] * Z[1] * Z[1] + p[5]) / p[0];

				J[0][1] = -(p[3] * (numb)4.0 * Z[1] * Z[1] * Z[1] * (Z[0] - p[4])) / p[0];
				J[0][2] = -(p[1] * (numb)3.0 * Z[2] * Z[2] * Z[3] * (Z[0] - p[2])) / p[0];
				J[0][3] = -(p[1] * Z[2] * Z[2] * Z[2] * (Z[0] - p[2])) / p[0];
				J[0][4] = (p[8] * (numb)4.0 * p[11] * (fmod(floor(((numb)4.0 * p[11] * (Z[4] - p[9]) + (numb)1.0) / (numb)2.0), (numb)2.0) < (numb)1.0 ? (numb)1.0 : -(numb)1.0)) / p[0];

				J[1][0] = (-(numb)0.01 * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)10.0 - Z[0]) * exp(((numb)10.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[1]) - (-(numb)0.0015625 * exp(-Z[0] / (numb)80.0)) * Z[1];
				J[1][1] = -((numb)0.01 * (((numb)10.0 - Z[0]) / (exp(((numb)10.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)0.125 * exp(-Z[0] / (numb)80.0));

				J[2][0] = (-(numb)0.1 * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0 - (numb)0.1 * ((numb)25.0 - Z[0]) * exp(((numb)25.0 - Z[0]) / (numb)10.0)) / ((exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0) * (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0))) * ((numb)1.0 - Z[2]) - (-((numb)4.0 / (numb)18.0) * exp(-Z[0] / (numb)18.0)) * Z[2];
				J[2][2] = -((numb)0.1 * (((numb)25.0 - Z[0]) / (exp(((numb)25.0 - Z[0]) / (numb)10.0) - (numb)1.0)) + (numb)4.0 * exp(-Z[0] / (numb)18.0));

				J[3][0] = (-(numb)0.0035 * exp(-Z[0] / (numb)20.0)) * ((numb)1.0 - Z[3]) - ((numb)0.1 * exp(((numb)30.0 - Z[0]) / (numb)10.0) / ((exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0) * (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0))) * Z[3];
				J[3][3] = -((numb)0.07 * exp(-Z[0] / (numb)20.0) + (numb)1.0 / (exp(((numb)30.0 - Z[0]) / (numb)10.0) + (numb)1.0));

				for (i = 0; i < N; i++) {
					for (j = 0; j < N; j++) {
						if (i == j)
							W[i][j] = (numb)1.0 - (numb)0.5 * h * J[i][j];
						else
							W[i][j] = -(numb)0.5 * h * J[i][j];
					}
				}

				for (i = 0; i < N; i++)
					a[i] = (numb)0.5 * (v[i] + Z[i]);

				I = p[7] + p[8] * (((numb)4.0 * p[11] * (a[4] - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (a[4] - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (a[4] - p[9]) + (numb)1.0) / (numb)2.0)));

				I_Na = (a[2] * a[2] * a[2]) * a[3] * p[1] * (a[0] - p[2]);
				I_K = (a[1] * a[1] * a[1] * a[1]) * p[3] * (a[0] - p[4]);
				I_L = p[5] * (a[0] - p[6]);

				W[0][5] = v[0] - Z[0] + h * ((I - I_Na - I_K - I_L) / p[0]);

				alpha_n = (numb)0.01 * (((numb)10.0 - a[0]) / (exp(((numb)10.0 - a[0]) / (numb)10.0) - (numb)1.0));
				beta_n = (numb)0.125 * exp(-a[0] / (numb)80.0);
				alpha_m = (numb)0.1 * (((numb)25.0 - a[0]) / (exp(((numb)25.0 - a[0]) / (numb)10.0) - (numb)1.0));
				beta_m = (numb)4.0 * exp(-a[0] / (numb)18.0);
				alpha_h = (numb)0.07 * exp(-a[0] / (numb)20.0);
				beta_h = (numb)1.0 / (exp(((numb)30.0 - a[0]) / (numb)10.0) + (numb)1.0);

				W[1][N] = v[1] - Z[1] + h * (alpha_n * ((numb)1.0 - a[1]) - beta_n * a[1]);
				W[2][N] = v[2] - Z[2] + h * (alpha_m * ((numb)1.0 - a[2]) - beta_m * a[2]);
				W[3][N] = v[3] - Z[3] + h * (alpha_h * ((numb)1.0 - a[3]) - beta_h * a[3]);
				W[4][N] = v[4] - Z[4] + h * ((numb)1.0);

				int k;
				numb b, d;

				for (k = 0; k <= N - 2; k++) {

					int l = k;

					for (i = k + 1; i <= N - 1; i++) {
						if (fabs(W[i][k]) > fabs(W[l][k])) {
							l = i;
						}
					}
					if (l != k) {
						for (j = 0; j <= N; j++) {
							if ((j == 0) || (j >= k)) {
								b = W[k][j];
								W[k][j] = W[l][j];
								W[l][j] = b;
							}
						}
					}

					d = (numb)1.0 / W[k][k];
					for (i = (k + 1); i <= (N - 1); i++) {
						if (W[i][k] == (numb)0.0) {
							continue;
						}
						b = W[i][k] * d;
						for (j = k; j <= N; j++) {
							if (W[k][j] != (numb)0.0) {
								W[i][j] = W[i][j] - b * W[k][j];
							}
						}
					}
				}

				for (i = N; i >= 2; i--) {
					for (j = 1; j <= i - 1; j++) {
						b = W[i - j - 1][i - 1] / W[i - 1][i - 1];
						W[i - j - 1][N] = W[i - j - 1][N] - b * W[i - 1][N];
					}
					W[i - 1][N] = W[i - 1][N] / W[i - 1][i - 1];
				}
				W[0][N] = W[0][N] / W[0][0];

				for (i = 0; i < N; i++)
					Zn[i] = Z[i] + W[i][N];

				dz = (numb)0.0;
				for (i = 0; i < N; i++) {
					numb diff = Zn[i] - Z[i];
					dz += diff * diff;
				}
				dz = sqrt(dz);

				for (i = 0; i < N; i++)
					Z[i] = Zn[i];

				nnewt++;
			}

			Vnext(v) = Zn[0];
			Vnext(n) = Zn[1];
			Vnext(m) = Zn[2];
			Vnext(h) = Zn[3];
			Vnext(t) = Zn[4];

			Vnext(i) = p[7] + p[8] * (((numb)4.0 * p[11] * (Vnext(t) - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)));
		}

		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb k[N][4];
			numb a[N];
			numb I;
			int i, j;

			for (i = 0; i < N; i++)
				a[i] = v[i];

			for (j = 0; j < 4; j++) {

				I = p[7] + p[8] * (((numb)4.0 * p[11] * (a[4] - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (a[4] - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (a[4] - p[9]) + (numb)1.0) / (numb)2.0)));

				numb alpha_n = (numb)0.01 * (((numb)10.0 - a[0]) / (exp(((numb)10.0 - a[0]) / (numb)10.0) - (numb)1.0));
				numb beta_n = (numb)0.125 * exp(-a[0] / (numb)80.0);

				numb alpha_m = (numb)0.1 * (((numb)25.0 - a[0]) / (exp(((numb)25.0 - a[0]) / (numb)10.0) - (numb)1.0));
				numb beta_m = (numb)4.0 * exp(-a[0] / (numb)18.0);

				numb alpha_h = (numb)0.07 * exp(-a[0] / (numb)20.0);
				numb beta_h = (numb)1.0 / (exp(((numb)30.0 - a[0]) / (numb)10.0) + (numb)1.0);

				numb I_Na = a[2] * a[2] * a[2] * a[3] * p[1] * (a[0] - p[2]);
				numb I_K = a[1] * a[1] * a[1] * a[1] * p[3] * (a[0] - p[4]);
				numb I_L = p[5] * (a[0] - p[6]);

				k[0][j] = (I - I_Na - I_K - I_L) / p[0];
				k[1][j] = alpha_n * ((numb)1.0 - a[1]) - beta_n * a[1];
				k[2][j] = alpha_m * ((numb)1.0 - a[2]) - beta_m * a[2];
				k[3][j] = alpha_h * ((numb)1.0 - a[3]) - beta_h * a[3];
				k[4][j] = (numb)1.0;

				if (j == 3) {

					Vnext(v) = v[0] + h * (k[0][0] + (numb)2.0 * k[0][1] + (numb)2.0 * k[0][2] + k[0][3]) / (numb)6.0;
					Vnext(n) = v[1] + h * (k[1][0] + (numb)2.0 * k[1][1] + (numb)2.0 * k[1][2] + k[1][3]) / (numb)6.0;
					Vnext(m) = v[2] + h * (k[2][0] + (numb)2.0 * k[2][1] + (numb)2.0 * k[2][2] + k[2][3]) / (numb)6.0;
					Vnext(h) = v[3] + h * (k[3][0] + (numb)2.0 * k[3][1] + (numb)2.0 * k[3][2] + k[3][3]) / (numb)6.0;
					Vnext(t) = v[4] + h * (k[4][0] + (numb)2.0 * k[4][1] + (numb)2.0 * k[4][2] + k[4][3]) / (numb)6.0;
				}
				else if (j == 2) {

					for (i = 0; i < N; i++)
						a[i] = v[i] + h * k[i][j];
				}
				else {

					for (i = 0; i < N; i++)
						a[i] = v[i] + (numb)0.5 * h * k[i][j];
				}
			}

			Vnext(i) = p[7] + p[8] * (((numb)4.0 * p[11] * (Vnext(t) - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)));
		}

		ifMETHOD(P(method), ExplicitDormandPrince8)
		{
			const numb M[13][12] = { {(numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.05555555555556, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.02083333333333, (numb)0.0625, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.03125, (numb)0.0, (numb)0.09375, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.3125, (numb)0.0, -(numb)1.171875, (numb)1.171875, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.0375, (numb)0.0, (numb)0.0, (numb)0.1875, (numb)0.15, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.04791013711111, (numb)0.0, (numb)0.0, (numb)0.1122487127778, -(numb)0.02550567377778, (numb)0.01284682388889, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.01691798978729, (numb)0.0, (numb)0.0, (numb)0.387848278486, (numb)0.0359773698515, (numb)0.1969702142157, -(numb)0.1727138523405, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.06909575335919, (numb)0.0, (numb)0.0, -(numb)0.6342479767289, -(numb)0.1611975752246, (numb)0.1386503094588, (numb)0.9409286140358, (numb)0.2116363264819, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0},
								{(numb)0.183556996839, (numb)0.0, (numb)0.0, -(numb)2.468768084316, -(numb)0.2912868878163, -(numb)0.02647302023312, (numb)2.847838764193, (numb)0.2813873314699, (numb)0.1237448998633, (numb)0.0, (numb)0.0, (numb)0.0},
								{-(numb)1.215424817396, (numb)0.0, (numb)0.0, (numb)16.67260866595, (numb)0.9157418284168, -(numb)6.056605804357, -(numb)16.00357359416, (numb)14.8493030863, -(numb)13.37157573529, (numb)5.13418264818, (numb)0.0, (numb)0.0},
								{(numb)0.2588609164383, (numb)0.0, (numb)0.0, -(numb)4.774485785489, -(numb)0.435093013777, -(numb)3.049483332072, (numb)5.577920039936, (numb)6.155831589861, -(numb)5.062104586737, (numb)2.193926173181, (numb)0.1346279986593, (numb)0.0},
								{(numb)0.8224275996265, (numb)0.0, (numb)0.0, -(numb)11.65867325728, -(numb)0.7576221166909, (numb)0.7139735881596, (numb)12.07577498689, -(numb)2.12765911392, (numb)1.990166207049, -(numb)0.234286471544, (numb)0.1758985777079, (numb)0.0} };

			const numb b[13] = { (numb)0.04174749114153, (numb)0.0, (numb)0.0, (numb)0.0, (numb)0.0, -(numb)0.05545232861124, (numb)0.2393128072012, (numb)0.7035106694034, -(numb)0.7597596138145, (numb)0.6605630309223, (numb)0.1581874825101, -(numb)0.2381095387529, (numb)0.25 };
			numb y[N], X1[N], X2[N];
			numb k[N][13];
			int i, j, l;
			numb I;

			for (i = 0; i < N; i++)
				X1[i] = v[i];

			for (i = 0; i < 13; i++) {

				I = p[7] + p[8] * (((numb)4.0 * p[11] * (X1[4] - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (X1[4] - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (X1[4] - p[9]) + (numb)1.0) / (numb)2.0)));

				numb alpha_n = (numb)0.01 * (((numb)10.0 - X1[0]) / (exp(((numb)10.0 - X1[0]) / (numb)10.0) - (numb)1.0));
				numb beta_n = (numb)0.125 * exp(-X1[0] / (numb)80.0);

				numb alpha_m = (numb)0.1 * (((numb)25.0 - X1[0]) / (exp(((numb)25.0 - X1[0]) / (numb)10.0) - (numb)1.0));
				numb beta_m = (numb)4.0 * exp(-X1[0] / (numb)18.0);

				numb alpha_h = (numb)0.07 * exp(-X1[0] / (numb)20.0);
				numb beta_h = (numb)1.0 / (exp(((numb)30.0 - X1[0]) / (numb)10.0) + (numb)1.0);

				numb I_Na = X1[2] * X1[2] * X1[2] * X1[3] * p[1] * (X1[0] - p[2]);
				numb I_K = X1[1] * X1[1] * X1[1] * X1[1] * p[3] * (X1[0] - p[4]);
				numb I_L = p[5] * (X1[0] - p[6]);

				k[0][j] = (I - I_Na - I_K - I_L) / p[0];
				k[1][j] = alpha_n * ((numb)1.0 - X1[1]) - beta_n * X1[1];
				k[2][j] = alpha_m * ((numb)1.0 - X1[2]) - beta_m * X1[2];
				k[3][j] = alpha_h * ((numb)1.0 - X1[3]) - beta_h * X1[3];
				k[4][j] = (numb)1.0;

				for (l = 0; l < N; l++)
					X2[l] = 0;

				for (j = 0; j < i + 1; j++)
					for (l = 0; l < N; l++)
						X2[l] += M[i + 1][j] * k[l][j];

				for (l = 0; l < N; l++)
					X1[l] = v[l] + H * X2[l];
			}

			for (l = 0; l < N; l++)
				X2[l] = 0;

			for (i = 0; i < 13; i++)
				for (l = 0; l < N; l++)
					X2[l] += b[i] * k[l][i];

			for (l = 0; l < N; l++)
				y[l] = v[l] + H * X2[l];

			Vnext(v) = y[0];
			Vnext(n) = y[1];
			Vnext(m) = y[2];
			Vnext(h) = y[3];
			Vnext(t) = y[4];

			Vnext(i) = p[7] + p[8] * (((numb)4.0 * p[11] * (Vnext(t) - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)));
		}

		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = (numb)0.5 * h - p[12];
			numb h2 = (numb)0.5 * h + p[12];

			numb I = p[7] + p[8] * (((numb)4.0 * p[11] * (v[4] - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (v[4] - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (v[4] - p[9]) + (numb)1.0) / (numb)2.0)));
			numb I_Na = (v[2] * v[2] * v[2]) * p[1] * v[3] * (v[0] - p[2]);
			numb I_K = (v[1] * v[1] * v[1] * v[1]) * p[3] * (v[0] - p[4]);
			numb I_L = p[5] * (v[0] - p[6]);

			numb vmp = v[0] + h1 * ((I - I_K - I_Na - I_L) / p[0]);

			numb alpha_n = (numb)0.01 * (((numb)10.0 - vmp) / (exp(((numb)10.0 - vmp) / (numb)10.0) - (numb)1.0));
			numb beta_n = (numb)0.125 * exp(-vmp / (numb)80.0);
			numb alpha_m = (numb)0.1 * (((numb)25.0 - vmp) / (exp(((numb)25.0 - vmp) / (numb)10.0) - (numb)1.0));
			numb beta_m = (numb)4.0 * exp(-vmp / (numb)18.0);
			numb alpha_h = (numb)0.07 * exp(-vmp / (numb)20.0);
			numb beta_h = (numb)1.0 / (exp(((numb)30.0 - vmp) / (numb)10.0) + (numb)1.0);

			numb nmp = v[1] + h1 * (alpha_n * ((numb)1.0 - v[1]) - beta_n * v[1]);
			numb mmp = v[2] + h1 * (alpha_m * ((numb)1.0 - v[2]) - beta_m * v[2]);
			numb hmp = v[3] + h1 * (alpha_h * ((numb)1.0 - v[3]) - beta_h * v[3]);

			Vnext(t) = V(t) + h;

			Vnext(h) = (hmp + h2 * alpha_h) / ((numb)1.0 + h2 * (alpha_h + beta_h));
			Vnext(m) = (mmp + h2 * alpha_m) / ((numb)1.0 + h2 * (alpha_m + beta_m));
			Vnext(n) = (nmp + h2 * alpha_n) / ((numb)1.0 + h2 * (alpha_n + beta_n));

			Vnext(i) = p[7] + p[8] * (((numb)4.0 * p[11] * (Vnext(t) - p[9]) - (numb)2.0 * floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)) * pow(-(numb)1.0, floor(((numb)4.0 * p[11] * (Vnext(t) - p[9]) + (numb)1.0) / (numb)2.0)));
			numb x_Na = (Vnext(m) * Vnext(m) * Vnext(m)) * p[1] * Vnext(h);
			numb x_K = (Vnext(n) * Vnext(n) * Vnext(n) * Vnext(n)) * p[3];

			Vnext(v) = (vmp + h2 * p[0] * (Vnext(i) + x_Na * p[2] + x_K * p[4] + p[5] * p[6])) / (1 + h2 * p[0] * (x_Na + x_K + p[5]));
		}
	}
}

#undef name