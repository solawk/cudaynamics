#include "main.h"
#include "resonate_and_fire.h"

#define name resonate_and_fire

namespace attributes
{
    enum variables { v, u, i, t };
    enum parameters { a, b, c, d, vpeak, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, method, COUNT };
	enum waveforms { square, sine, triangle };
    enum methods { ExplicitEuler, SemiExplicitEuler, ImplicitEuler, ExplicitMidpoint, ImplicitMidpoint, ExplicitRungeKutta4, ExplicitDormandPrince8, VariableSymmetryCD};
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

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters)
{
	const numb h = H;
	const int N = 3;
	const numb p[11] = { P(a), P(b), P(c), P(d), P(vpeak), P(Idc), P(Iamp), P(Ifreq), P(Idel), P(Idf), P(symmetry)};
	numb v[N] = { V(v), V(u), V(t) };

	ifSIGNAL(P(signal), square)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			Vnext(i) = p[5] + (fmod((v[2] - p[8]) > 0 ? (v[2] - p[8]) : (p[9] / p[7] + p[8] - v[2]), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);
			Vnext(t) = v[2] + h;
			Vnext(v) = v[0] + h * ((p[0] * v[1] + p[1] * v[0]) + Vnext(i));
			Vnext(u) = v[1] + h * ((p[1] * v[1] - p[0] * v[0]));

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}
		ifMETHOD(P(method), SemiExplicitEuler)
		{
			Vnext(i) = p[5] + (fmod((v[2] - p[8]) > 0 ? (v[2] - p[8]) : (p[9] / p[7] + p[8] - v[2]), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);
			Vnext(t) = v[2] + h;
			Vnext(v) = v[0] + h * ((p[0] * v[1] + p[1] * v[0]) + Vnext(i));
			Vnext(u) = v[1] + h * ((p[1] * v[1] - p[0] * Vnext(v)));

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}

		ifMETHOD(P(method), ImplicitEuler)
		{
			numb Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[0][2] = (numb)0.0;

			J[1][2] = (numb)0.0;

			J[2][0] = (numb)0.0;
			J[2][1] = (numb)0.0;
			J[2][2] = (numb)0.0;

			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = p[1];
				J[0][1] = p[0];

				J[1][0] = -p[0];
				J[1][1] = p[1];

				for (i = 0; i < N; i++) {
					for (j = 0; j < N; j++) {
						if (i == j)
							W[i][j] = (numb)1.0 - h * J[i][j];
						else
							W[i][j] = -h * J[i][j];
					}
				}

				I = p[5] + (fmod((Z[2] - p[8]) > 0 ? (Z[2] - p[8]) : (p[9] / p[7] + p[8] - Z[2]), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);

				W[0][N] = v[0] - Z[0] + h * ((p[0] * Z[1] + p[1] * Z[0]) + I);
				W[1][N] = v[1] - Z[1] + h * (p[1] * Z[1] - p[0] * v[0]);
				W[2][N] = v[2] - Z[2] + h * ((numb)1.0);

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
			Vnext(u) = Zn[1];
			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}

			Vnext(t) = Zn[2];
			Vnext(i) = p[5] + (fmod((Vnext(t) - p[8]) > 0 ? (Vnext(t) - p[8]) : (p[9] / p[7] + p[8] - Vnext(t)), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);
		}

		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = p[5] + (fmod((v[2] - p[8]) > 0 ? (v[2] - p[8]) : (p[9] / p[7] + p[8] - v[2]), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);
			numb tmp = v[2] + h * (numb)0.5;
			numb vmp = v[0] + h * (numb)0.5 * ((p[0] * v[1] + p[1] * v[0]) + imp);
			numb ump = v[1] + h * (numb)0.5 * ((p[1] * v[1] - p[0] * v[0]));


			Vnext(i) = p[5] + (fmod((tmp - p[8]) > 0 ? (tmp - p[8]) : (p[9] / p[7] + p[8] - tmp), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);
			Vnext(t) = v[2] + h;
			Vnext(v) = v[0] + h * ((p[0] * ump + p[1] * vmp) + Vnext(i));
			Vnext(u) = v[1] + h * ((p[1] * ump - p[0] * vmp));

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}

		ifMETHOD(P(method), ImplicitMidpoint)
		{
			numb a[N], Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[0][2] = (numb)0.0;

			J[1][2] = (numb)0.0;

			J[2][0] = (numb)0.0;
			J[2][1] = (numb)0.0;
			J[2][2] = (numb)0.0;


			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = p[1];
				J[0][1] = p[0];

				J[1][0] = -p[0];
				J[1][1] = p[1];

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

				I = p[5] + (fmod((a[2] - p[8]) > 0 ? (a[2] - p[8]) : (p[9] / p[7] + p[8] - a[2]), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);

				W[0][N] = v[0] - Z[0] + h * ((p[0] * a[1] + p[1] * a[0]) + I);
				W[1][N] = v[1] - Z[1] + h * (p[1] * a[1] - p[0] * a[0]);
				W[2][N] = v[2] - Z[2] + h * ((numb)1.0);

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
			Vnext(u) = Zn[1];
			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}

			Vnext(t) = Zn[2];
			Vnext(i) = p[5] + (fmod((Vnext(t) - p[8]) > 0 ? (Vnext(t) - p[8]) : (p[9] / p[7] + p[8] - Vnext(t)), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);
		}

		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb k[2][4];
			numb a[2];
			numb I; numb tmp;
			tmp = v[2];
			int i, j;

			for (i = 0; i < 2; i++)
				a[i] = v[i];

			for (j = 0; j < 4; j++) {
				I = p[5] + (fmod((tmp - p[8]) > 0 ? (tmp - p[8]) : (p[9] / p[7] + p[8] - tmp), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);
				k[0][j] = ((p[0] * a[1] + p[1] * a[0]) + I);
				k[1][j] = ((p[1] * a[1] - p[0] * a[0]));


				if (j == 3) {

					Vnext(v) = v[0] + h * (k[0][0] + (numb)2.0 * k[0][1] + (numb)2.0 * k[0][2] + k[0][3]) / (numb)6.0;
					Vnext(u) = v[1] + h * (k[1][0] + (numb)2.0 * k[1][1] + (numb)2.0 * k[1][2] + k[1][3]) / (numb)6.0;
					Vnext(t) = tmp;
					Vnext(i) = I;
					if (Vnext(v) >= p[4])
					{
						Vnext(v) = p[2];
						Vnext(u) = p[3];
					}
				}
				else if (j == 2) {
					for (i = 0; i < 2; i++)
						a[i] = v[i] + h * k[i][j];
					tmp = v[2] + h;
				}
				else {
					for (i = 0; i < 2; i++)
						a[i] = v[i] + (numb)0.5 * h * k[i][j];
					tmp = v[2] + h * (numb)0.5;
				}
				
			}
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
			int i = 0, j = 0, l = 0;
			numb I;

			for (i = 0; i < N; i++)
				X1[i] = v[i];

			for (i = 0; i < 13; i++) {
				I = p[5] + (fmod((X1[2] - p[8]) > 0 ? (X1[2] - p[8]) : (p[9] / p[7] + p[8] - X1[2]), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);


				k[0][i] = ((p[0] * X1[1] + p[1] * X1[0]) + I);
				k[1][i] = ((p[1] * X1[1] - p[0] * X1[0]));
				k[2][i] = (numb)1.0;

				for (l = 0; l < N; l++)
					X2[l] = 0;

				for (j = 0; j < i + 1; j++)
					for (l = 0; l < N; l++)
						X2[l] += M[i + 1][j] * k[l][j];

				for (l = 0; l < N; l++)
					X1[l] = v[l] + H * X2[l];

				if (X1[0] >= p[4])
				{
					X1[0] = p[2];
					X1[1] = p[3];
				}
			}

			for (l = 0; l < N; l++)
				X2[l] = 0;

			for (i = 0; i < 13; i++)
				for (l = 0; l < N; l++)
					X2[l] += b[i] * k[l][i];

			for (l = 0; l < N; l++)
				y[l] = v[l] + H * X2[l];

			Vnext(v) = y[0];
			Vnext(u) = y[1];
			Vnext(t) = y[2];
			Vnext(i) = p[5] + (fmod((Vnext(t) - p[8]) > 0 ? (Vnext(t) - p[8]) : (p[9] / p[7] + p[8] - Vnext(t)), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);
			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}

		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = (numb)0.5 * h - p[10];
			numb h2 = (numb)0.5 * h + p[10];

			numb I = p[5] + (fmod((v[2] - p[8]) > 0 ? (v[2] - p[8]) : (p[9] / p[7] + p[8] - v[2]), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);

			numb vmp = v[0] + h1 * ((p[0] * v[1] + p[1] * v[0]) + I);
			numb ump = v[1] + h1 * ((p[1] * v[1] - p[0] * vmp));


			Vnext(t) = v[2] + h;
			
			Vnext(i) = p[5] + (fmod((Vnext(t) - p[8]) > 0 ? (Vnext(t) - p[8]) : (p[9] / p[7] + p[8] - Vnext(t)), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);

			Vnext(u) = (ump - h2 * p[0] * vmp) / ((numb)1.0 - h2 * p[1]);
			Vnext(v) = (vmp + h2 * p[0] * Vnext(u) + h2 * Vnext(i)) / ((numb)1.0 - h2 * p[1]);

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}
	}

	ifSIGNAL(P(signal), sine)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			Vnext(i) = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (v[2] - p[8]));
			Vnext(t) = v[2] + h;
			Vnext(v) = v[0] + h * ((p[0] * v[1] + p[1] * v[0]) + Vnext(i));
			Vnext(u) = v[1] + h * ((p[1] * v[1] - p[0] * v[0]));

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}
		ifMETHOD(P(method), SemiExplicitEuler)
		{
			Vnext(i) = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (v[2] - p[8]));
			Vnext(t) = v[2] + h;
			Vnext(v) = v[0] + h * ((p[0] * v[1] + p[1] * v[0]) + Vnext(i));
			Vnext(u) = v[1] + h * ((p[1] * v[1] - p[0] * Vnext(v)));

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}

		ifMETHOD(P(method), ImplicitEuler)
		{
			numb Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[0][2] = (numb)0.0;

			J[1][2] = (numb)0.0;

			J[2][0] = (numb)0.0;
			J[2][1] = (numb)0.0;
			J[2][2] = (numb)0.0;

			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = p[1];
				J[0][1] = p[0];

				J[1][0] = -p[0];
				J[1][1] = p[1];

				for (i = 0; i < N; i++) {
					for (j = 0; j < N; j++) {
						if (i == j)
							W[i][j] = (numb)1.0 - h * J[i][j];
						else
							W[i][j] = -h * J[i][j];
					}
				}

				I = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (Z[2] - p[8]));

				W[0][N] = v[0] - Z[0] + h * ((p[0] * Z[1] + p[1] * Z[0]) + I);
				W[1][N] = v[1] - Z[1] + h * (p[1] * Z[1] - p[0] * v[0]);
				W[2][N] = v[2] - Z[2] + h * ((numb)1.0);

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
			Vnext(u) = Zn[1];
			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}

			Vnext(t) = Zn[2];
			Vnext(i) = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (Vnext(t) - p[8]));
		}

		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (v[2] - p[8]));
			numb tmp = v[2] + h * (numb)0.5;
			numb vmp = v[0] + h * (numb)0.5 * ((p[0] * v[1] + p[1] * v[0]) + imp);
			numb ump = v[1] + h * (numb)0.5 * ((p[1] * v[1] - p[0] * v[0]));


			Vnext(i) = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (tmp- p[8]));
			Vnext(t) = v[2] + h;
			Vnext(v) = v[0] + h * ((p[0] * ump + p[1] * vmp) + Vnext(i));
			Vnext(u) = v[1] + h * ((p[1] * ump - p[0] * vmp));

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}

		ifMETHOD(P(method), ImplicitMidpoint)
		{
			numb a[N], Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[0][2] = (numb)0.0;

			J[1][2] = (numb)0.0;

			J[2][0] = (numb)0.0;
			J[2][1] = (numb)0.0;
			J[2][2] = (numb)0.0;

			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = p[1];
				J[0][1] = p[0];

				J[1][0] = -p[0];
				J[1][1] = p[1];

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

				I = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (a[2] - p[8]));

				W[0][N] = v[0] - Z[0] + h * ((p[0] * a[1] + p[1] * a[0]) + I);
				W[1][N] = v[1] - Z[1] + h * (p[1] * a[1] - p[0] * a[0]);
				W[2][N] = v[2] - Z[2] + h * ((numb)1.0);

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
			Vnext(u) = Zn[1];
			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}

			Vnext(t) = Zn[2];
			Vnext(i) = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (Vnext(t) - p[8]));
		}

		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb k[2][4];
			numb a[2];
			numb I; numb tmp;
			tmp = v[2];
			int i, j;

			for (i = 0; i < 2; i++)
				a[i] = v[i];

			for (j = 0; j < 4; j++) {
				I = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (tmp - p[8]));
				k[0][j] = ((p[0] * a[1] + p[1] * a[0]) + I);
				k[1][j] = ((p[1] * a[1] - p[0] * a[0]));


				if (j == 3) {

					Vnext(v) = v[0] + h * (k[0][0] + (numb)2.0 * k[0][1] + (numb)2.0 * k[0][2] + k[0][3]) / (numb)6.0;
					Vnext(u) = v[1] + h * (k[1][0] + (numb)2.0 * k[1][1] + (numb)2.0 * k[1][2] + k[1][3]) / (numb)6.0;
					Vnext(t) = tmp;
					Vnext(i) = I;
					if (Vnext(v) >= p[4])
					{
						Vnext(v) = p[2];
						Vnext(u) = p[3];
					}
				}
				else if (j == 2) {
					for (i = 0; i < 2; i++)
						a[i] = v[i] + h * k[i][j];
					tmp = v[2] + h;
				}
				else {
					for (i = 0; i < 2; i++)
						a[i] = v[i] + (numb)0.5 * h * k[i][j];
					tmp = v[2] + h * (numb)0.5;
				}

			}
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
			int i = 0, j = 0, l = 0;
			numb I;

			for (i = 0; i < N; i++)
				X1[i] = v[i];

			for (i = 0; i < 13; i++) {
				I = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (X1[2] - p[8]));


				k[0][i] = ((p[0] * X1[1] + p[1] * X1[0]) + I);
				k[1][i] = ((p[1] * X1[1] - p[0] * X1[0]));
				k[2][i] = (numb)1.0;

				for (l = 0; l < N; l++)
					X2[l] = 0;

				for (j = 0; j < i + 1; j++)
					for (l = 0; l < N; l++)
						X2[l] += M[i + 1][j] * k[l][j];

				for (l = 0; l < N; l++)
					X1[l] = v[l] + H * X2[l];

				if (X1[0] >= p[4])
				{
					X1[0] = p[2];
					X1[1] = p[3];
				}
			}

			for (l = 0; l < N; l++)
				X2[l] = 0;

			for (i = 0; i < 13; i++)
				for (l = 0; l < N; l++)
					X2[l] += b[i] * k[l][i];

			for (l = 0; l < N; l++)
				y[l] = v[l] + H * X2[l];

			Vnext(v) = y[0];
			Vnext(u) = y[1];
			Vnext(t) = y[2];
			Vnext(i) = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (Vnext(t) - p[8]));
			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}

		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = (numb)0.5 * h - p[10];
			numb h2 = (numb)0.5 * h + p[10];

			numb I = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (v[2] - p[8]));

			numb vmp = v[0] + h1 * ((p[0] * v[1] + p[1] * v[0]) + I);
			numb ump = v[1] + h1 * ((p[1] * v[1] - p[0] * vmp));


			Vnext(t) = v[2] + h;

			Vnext(i) = p[5] + p[6] * sin((numb)2.0 * (numb)3.141592653589793 * p[7] * (Vnext(t) - p[8]));

			Vnext(u) = (ump - h2 * p[0] * vmp) / ((numb)1.0 - h2 * p[1]);
			Vnext(v) = (vmp + h2 * p[0] * Vnext(u) + h2 * Vnext(i)) / ((numb)1.0 - h2 * p[1]);

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}
	}

	ifSIGNAL(P(signal), triangle)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{

			Vnext(i) = p[5] + p[6] * (((numb)4.0 * p[7] * (v[2] - p[8]) - (numb)2.0 * floor(((numb)4.0 * p[8] * (v[2] - p[8]) + (numb)1.0) / (numb)2.0)) * ((int)floor(((numb)4.0 * p[7] * (v[2] - p[8]) + (numb)1.0) / (numb)2.0) % 2 == 0 ? (numb)1.0 : (numb)-1.0));
			Vnext(t) = v[2] + h;
			Vnext(v) = v[0] + h * ((p[0] * v[1] + p[1] * v[0]) + Vnext(i));
			Vnext(u) = v[1] + h * ((p[1] * v[1] - p[0] * v[0]));

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}
		ifMETHOD(P(method), SemiExplicitEuler)
		{
			Vnext(i) = p[5] + p[6] * (((numb)4.0 * p[7] * (v[2] - p[8]) - (numb)2.0 * floor(((numb)4.0 * p[8] * (v[2] - p[8]) + (numb)1.0) / (numb)2.0)) * ((int)floor(((numb)4.0 * p[7] * (v[2] - p[8]) + (numb)1.0) / (numb)2.0) % 2 == 0 ? (numb)1.0 : (numb)-1.0));
			Vnext(t) = v[2] + h;
			Vnext(v) = v[0] + h * ((p[0] * v[1] + p[1] * v[0]) + Vnext(i));
			Vnext(u) = v[1] + h * ((p[1] * v[1] - p[0] * Vnext(v)));

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}

		ifMETHOD(P(method), ImplicitEuler)
		{
			numb Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[0][2] = (numb)0.0;

			J[1][2] = (numb)0.0;


			J[2][0] = (numb)0.0;
			J[2][1] = (numb)0.0;
			J[2][2] = (numb)0.0;

			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = p[1];
				J[0][1] = p[0];

				J[1][0] = -p[0];
				J[1][1] = p[1];

				for (i = 0; i < N; i++) {
					for (j = 0; j < N; j++) {
						if (i == j)
							W[i][j] = (numb)1.0 - h * J[i][j];
						else
							W[i][j] = -h * J[i][j];
					}
				}

				I = p[5] + p[6] * (((numb)4.0 * p[7] * (Z[2] - p[8]) - (numb)2.0 * floor(((numb)4.0 * p[8] * (Z[2] - p[8]) + (numb)1.0) / (numb)2.0)) * ((int)floor(((numb)4.0 * p[7] * (Z[2] - p[8]) + (numb)1.0) / (numb)2.0) % 2 == 0 ? (numb)1.0 : (numb)-1.0));

				W[0][N] = v[0] - Z[0] + h * ((p[0] * Z[1] + p[1] * Z[0]) + I);
				W[1][N] = v[1] - Z[1] + h * (p[1] * Z[1] - p[0] * v[0]);
				W[2][N] = v[2] - Z[2] + h * ((numb)1.0);

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
			Vnext(u) = Zn[1];
			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}

			Vnext(t) = Zn[2];
			Vnext(i) = p[5] + p[6] * (((numb)4.0 * p[7] * (Vnext(t) - p[8]) - (numb)2.0 * floor(((numb)4.0 * p[8] * (Vnext(t) - p[8]) + (numb)1.0) / (numb)2.0)) * ((int)floor(((numb)4.0 * p[7] * (Vnext(t) - p[8]) + (numb)1.0) / (numb)2.0) % 2 == 0 ? (numb)1.0 : (numb)-1.0));
		}

		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb imp = p[5] + p[6] * (((numb)4.0 * p[7] * (v[2] - p[8]) - (numb)2.0 * floor(((numb)4.0 * p[8] * (v[2] - p[8]) + (numb)1.0) / (numb)2.0)) * ((int)floor(((numb)4.0 * p[7] * (v[2] - p[8]) + (numb)1.0) / (numb)2.0) % 2 == 0 ? (numb)1.0 : (numb)-1.0));
			numb tmp = v[2] + h * (numb)0.5;
			numb vmp = v[0] + h * (numb)0.5 * ((p[0] * v[1] + p[1] * v[0]) + imp);
			numb ump = v[1] + h * (numb)0.5 * ((p[1] * v[1] - p[0] * v[0]));


			Vnext(i) = p[5] + p[6] * (((numb)4.0 * p[7] * (tmp - p[8]) - (numb)2.0 * floor(((numb)4.0 * p[8] * (tmp - p[8]) + (numb)1.0) / (numb)2.0)) * ((int)floor(((numb)4.0 * p[7] * (tmp - p[8]) + (numb)1.0) / (numb)2.0) % 2 == 0 ? (numb)1.0 : (numb)-1.0));
			Vnext(t) = v[2] + h;
			Vnext(v) = v[0] + h * ((p[0] * ump + p[1] * vmp) + Vnext(i));
			Vnext(u) = v[1] + h * ((p[1] * ump - p[0] * vmp));

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}

		ifMETHOD(P(method), ImplicitMidpoint)
		{
			numb a[N], Z[N], Zn[N], J[N][N], W[N][N + 1];
			numb tol = (numb)1e-14, dz = (numb)2e-13;
			numb I;
			int nnewtmax = 8, nnewt = 0;
			int i, j;

			J[0][2] = (numb)0.0;

			J[1][2] = (numb)0.0;

			J[2][0] = (numb)0.0;
			J[2][1] = (numb)0.0;
			J[2][2] = (numb)0.0;

			for (i = 0; i < N; i++)
				Z[i] = v[i];

			while ((dz > tol) && (nnewt < nnewtmax)) {

				J[0][0] = p[1];
				J[0][1] = p[0];

				J[1][0] = -p[0];
				J[1][1] = p[1];

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

				I = p[5] + p[6] * (((numb)4.0 * p[7] * (a[2] - p[8]) - (numb)2.0 * floor(((numb)4.0 * p[8] * (a[2] - p[8]) + (numb)1.0) / (numb)2.0)) * ((int)floor(((numb)4.0 * p[7] * (a[2] - p[8]) + (numb)1.0) / (numb)2.0) % 2 == 0 ? (numb)1.0 : (numb)-1.0));

				W[0][N] = v[0] - Z[0] + h * ((p[0] * a[1] + p[1] * a[0]) + I);
				W[1][N] = v[1] - Z[1] + h * (p[1] * a[1] - p[0] * a[0]);
				W[2][N] = v[2] - Z[2] + h * ((numb)1.0);

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
			Vnext(u) = Zn[1];
			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}

			Vnext(t) = Zn[2];
			Vnext(i) = p[5] + p[6] * (((numb)4.0 * p[7] * (Vnext(t) - p[8]) - (numb)2.0 * floor(((numb)4.0 * p[8] * (Vnext(t) - p[8]) + (numb)1.0) / (numb)2.0)) * ((int)floor(((numb)4.0 * p[7] * (Vnext(t) - p[8]) + (numb)1.0) / (numb)2.0) % 2 == 0 ? (numb)1.0 : (numb)-1.0));
		}

		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb k[2][4];
			numb a[2];
			numb I; numb tmp;
			tmp = v[2];
			int i, j;

			for (i = 0; i < 2; i++)
				a[i] = v[i];

			for (j = 0; j < 4; j++) {
				I = Vnext(i) = p[5] + p[6] * (((numb)4.0 * p[7] * (tmp - p[8]) - (numb)2.0 * floor(((numb)4.0 * p[8] * (tmp - p[8]) + (numb)1.0) / (numb)2.0)) * ((int)floor(((numb)4.0 * p[7] * (tmp - p[8]) + (numb)1.0) / (numb)2.0) % 2 == 0 ? (numb)1.0 : (numb)-1.0));
				k[0][j] = ((p[0] * a[1] + p[1] * a[0]) + I);
				k[1][j] = ((p[1] * a[1] - p[0] * a[0]));


				if (j == 3) {

					Vnext(v) = v[0] + h * (k[0][0] + (numb)2.0 * k[0][1] + (numb)2.0 * k[0][2] + k[0][3]) / (numb)6.0;
					Vnext(u) = v[1] + h * (k[1][0] + (numb)2.0 * k[1][1] + (numb)2.0 * k[1][2] + k[1][3]) / (numb)6.0;
					Vnext(t) = tmp;
					Vnext(i) = I;
					if (Vnext(v) >= p[4])
					{
						Vnext(v) = p[2];
						Vnext(u) = p[3];
					}
				}
				else if (j == 2) {
					for (i = 0; i < 2; i++)
						a[i] = v[i] + h * k[i][j];
					tmp = v[2] + h;
				}
				else {
					for (i = 0; i < 2; i++)
						a[i] = v[i] + (numb)0.5 * h * k[i][j];
					tmp = v[2] + h * (numb)0.5;
				}

			}
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
			int i = 0, j = 0, l = 0;
			numb I;

			for (i = 0; i < N; i++)
				X1[i] = v[i];

			for (i = 0; i < 13; i++) {
				I = p[5] + p[6] * (((numb)4.0 * p[7] * (X1[2] - p[8]) - (numb)2.0 * floor(((numb)4.0 * p[8] * (X1[2] - p[8]) + (numb)1.0) / (numb)2.0)) * ((int)floor(((numb)4.0 * p[7] * (X1[2] - p[8]) + (numb)1.0) / (numb)2.0) % 2 == 0 ? (numb)1.0 : (numb)-1.0));


				k[0][i] = ((p[0] * X1[1] + p[1] * X1[0]) + I);
				k[1][i] = ((p[1] * X1[1] - p[0] * X1[0]));
				k[2][i] = (numb)1.0;

				for (l = 0; l < N; l++)
					X2[l] = 0;

				for (j = 0; j < i + 1; j++)
					for (l = 0; l < N; l++)
						X2[l] += M[i + 1][j] * k[l][j];

				for (l = 0; l < N; l++)
					X1[l] = v[l] + H * X2[l];

				if (X1[0] >= p[4])
				{
					X1[0] = p[2];
					X1[1] = p[3];
				}
			}

			for (l = 0; l < N; l++)
				X2[l] = 0;

			for (i = 0; i < 13; i++)
				for (l = 0; l < N; l++)
					X2[l] += b[i] * k[l][i];

			for (l = 0; l < N; l++)
				y[l] = v[l] + H * X2[l];

			Vnext(v) = y[0];
			Vnext(u) = y[1];
			Vnext(t) = y[2];
			Vnext(i) = p[5] + p[6] * (((numb)4.0 * p[7] * (Vnext(t) - p[8]) - (numb)2.0 * floor(((numb)4.0 * p[8] * (Vnext(t) - p[8]) + (numb)1.0) / (numb)2.0)) * ((int)floor(((numb)4.0 * p[7] * (Vnext(t) - p[8]) + (numb)1.0) / (numb)2.0) % 2 == 0 ? (numb)1.0 : (numb)-1.0));
			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}

		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = (numb)0.5 * h - p[10];
			numb h2 = (numb)0.5 * h + p[10];

			numb I = p[5] + (fmod((v[2] - p[8]) > 0 ? (v[2] - p[8]) : (p[9] / p[7] + p[8] - v[2]), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);

			numb vmp = v[0] + h1 * ((p[0] * v[1] + p[1] * v[0]) + I);
			numb ump = v[1] + h1 * ((p[1] * v[1] - p[0] * vmp));


			Vnext(t) = v[2] + h;

			Vnext(i) = p[5] + (fmod((Vnext(t) - p[8]) > 0 ? (Vnext(t) - p[8]) : (p[9] / p[7] + p[8] - Vnext(t)), 1 / p[7]) < p[9] / p[7] ? p[6] : (numb)0.0);

			Vnext(u) = (ump - h2 * p[0] * vmp) / ((numb)1.0 - h2 * p[1]);
			Vnext(v) = (vmp + h2 * p[0] * Vnext(u) + h2 * Vnext(i)) / ((numb)1.0 - h2 * p[1]);

			if (Vnext(v) >= p[4])
			{
				Vnext(v) = p[2];
				Vnext(u) = p[3];
			}
		}
	}
}
