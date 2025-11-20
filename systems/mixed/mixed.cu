#include "main.h"
#include "mixed.h"

#define name mixed

namespace attributes
{
    enum variables { v, i, x, n, t };
    enum parameters { k1, k2, k3, a, Nmax, Ron, Roff, Vdc, Vamp, Vfreq, Vdel, Vdf, symmetry, signal, method, COUNT };
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
			Vnext(v) = P(Vdc) + (fmodf((V(t) - P(Vdel)) > 0 ? (V(t) - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - V(t)), 1 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : 0.0f);
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			Vnext(x) = V(x) + H * (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * Vnext(i));

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			Vnext(n) = V(n) + H * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * Vnext(i));
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb vmp = P(Vdc) + (fmodf((V(t) - P(Vdel)) > 0 ? (V(t) - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - V(t)), 1 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : 0.0f);
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			numb xmp = V(x) + H * 0.5f * (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * imp);

			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb nmp = V(n) + H * 0.5f * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * imp);
			numb tmp = V(t) + H * 0.5f;

			Vnext(v) = P(Vdc) + (fmodf((tmp - P(Vdel)) > 0 ? (tmp - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - tmp), 1 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : 0.0f);
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			Vnext(x) = V(x) + H * (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * Vnext(i));

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			Vnext(n) = V(n) + H * (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * Vnext(i));
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb vmp = P(Vdc) + (fmodf((V(t) - P(Vdel)) > 0 ? (V(t) - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - V(t)), 1 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : 0.0f);
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			numb kx1 = (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * imp);
			numb xmp = V(x) + H * 0.5f * kx1;

			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb kn1 = (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * Vnext(i));
			numb nmp = V(n) + H * 0.5f * kn1;
			numb tmp = V(t) + H * 0.5f;

			vmp = P(Vdc) + (fmodf((tmp - P(Vdel)) > 0 ? (tmp - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - tmp), 1 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : 0.0f);
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			numb kx2 = (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * imp);
			xmp = V(x) + H * 0.5f * kx2;

			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb kn2 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * imp);
			nmp = V(n) + H * 0.5f * kn2;
			tmp = V(t) + H * 0.5f;

			vmp = P(Vdc) + (fmodf((tmp - P(Vdel)) > 0 ? (tmp - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - tmp), 1 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : 0.0f);
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			numb kx3 = (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * imp);
			xmp = V(x) + H * kx3;

			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb kn3 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * imp);
			nmp = V(n) + H * kn3;
			tmp = V(t) + H;

			Vnext(v) = P(Vdc) + (fmodf((tmp - P(Vdel)) > 0 ? (tmp - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - tmp), 1 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : 0.0f);
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			numb kx4 = (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * Vnext(i));
			Vnext(x) = V(x) + H * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			numb kn4 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * Vnext(i));
			Vnext(n) = V(n) + H * (kn1 + kn2 * 2 + kn3 * 2 + kn4) / 6;
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), VariableSymmetryCD)
		{

			numb h1 = 0.5 * H - P(symmetry);
			numb h2 = 0.5 * H + P(symmetry);

			Vnext(v) = P(Vdc) + (fmodf((V(t) - P(Vdel)) > 0 ? (V(t) - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - V(t)), 1 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : 0.0f);
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			Vnext(x) = V(x) + h1 * (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * Vnext(i));

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			Vnext(n) = V(n) + h1 * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * Vnext(i));
			Vnext(t) = V(t) + h1;

			/////

			Vnext(t) = Vnext(t) + h2;
			Vnext(v) = P(Vdc) + (fmodf((Vnext(t) - P(Vdel)) > 0 ? (Vnext(t) - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - Vnext(t)), 1 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : 0.0f);
			Vnext(i) = (Vnext(v) - (P(k1) * (Vnext(n) / (Vnext(x) + P(a))))) / ((P(Ron) * Vnext(x)) + (P(Roff) * (1 - Vnext(x))));

			Vnext(n) = Vnext(n) + h2 * (P(k3) * ((Vnext(n) >= -P(Nmax) && Vnext(n) <= P(Nmax)) ? 1 - abs(Vnext(n) / P(Nmax)) : 0) * Vnext(i));

			Vnext(x) = Vnext(x) + h2 * (P(k2) * ((Vnext(x) >= 0 && Vnext(x) <= 1) ? 1 : 0) * Vnext(i));


			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

		}
	}
	ifSIGNAL(P(signal), sine)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			Vnext(v) = P(Vdc) + P(Vamp) * sinf(2.0f * 3.141592f * P(Vfreq) * (V(t) - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			Vnext(x) = V(x) + H * (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * Vnext(i));

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			Vnext(n) = V(n) + H * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * Vnext(i));
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb vmp = P(Vdc) + P(Vamp) * sinf(2.0f * 3.141592f * P(Vfreq) * (V(t) - P(Vdel)));
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			numb xmp = V(x) + H * 0.5f * (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * imp);
			
			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb nmp = V(n) + H * 0.5f * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * imp);
			numb tmp = V(t) + H * 0.5f;

			Vnext(v) = P(Vdc) + P(Vamp) * sinf(2.0f * 3.141592f * P(Vfreq) * (tmp - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			Vnext(x) = V(x) + H * (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * Vnext(i));

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			Vnext(n) = V(n) + H * (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * Vnext(i));
			Vnext(t) = V(t) + H;
		} 
		ifMETHOD(P(method), ExplicitRungeKutta4)
		{ 
			numb vmp = P(Vdc) + P(Vamp) * sinf(2.0f * 3.141592f * P(Vfreq) * (V(t) - P(Vdel)));
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			numb kx1 = (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * imp);
			numb xmp = V(x) + H * 0.5f * kx1;

			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb kn1 = (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * Vnext(i));
			numb nmp = V(n) + H * 0.5f * kn1;
			numb tmp = V(t) + H * 0.5f;

			vmp = P(Vdc) + P(Vamp) * sinf(2.0f * 3.141592f * P(Vfreq) * (tmp - P(Vdel)));
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			numb kx2 = (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * imp);
			xmp = V(x) + H * 0.5f * kx2;

			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb kn2 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * imp);
			nmp = V(n) + H * 0.5f * kn2;
			tmp = V(t) + H * 0.5f;

			vmp = P(Vdc) + P(Vamp) * sinf(2.0f * 3.141592f * P(Vfreq) * (tmp - P(Vdel)));
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			numb kx3 = (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * imp);
			xmp = V(x) + H * kx3;

			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb kn3 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * imp);
			nmp = V(n) + H * kn3;
			tmp = V(t) + H;

			Vnext(v) = P(Vdc) + P(Vamp) * sinf(2.0f * 3.141592f * P(Vfreq) * (tmp - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			numb kx4 = (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * Vnext(i));
			Vnext(x) = V(x) + H * (kx1 + 2 * kx2 + 2 * kx3 + kx4)/6;

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			numb kn4 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * Vnext(i));
			Vnext(n) = V(n) + H * (kn1 + kn2 * 2 + kn3 * 2 + kn4)/6;
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), VariableSymmetryCD)
		{

			numb h1 = 0.5 * H - P(symmetry);
			numb h2 = 0.5 * H + P(symmetry);

			Vnext(v) = P(Vdc) + P(Vamp) * sinf(2.0f * 3.141592f * P(Vfreq) * (V(t) - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			Vnext(x) = V(x) + h1 * (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * Vnext(i));

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			Vnext(n) = V(n) + h1 * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * Vnext(i));
			Vnext(t) = V(t) + h1;

			/////

			Vnext(t) = Vnext(t) + h2;
			Vnext(v) = P(Vdc) + P(Vamp) * sinf(2.0f * 3.141592f * P(Vfreq) * (Vnext(t) - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (Vnext(n) / (Vnext(x) + P(a))))) / ((P(Ron) * Vnext(x)) + (P(Roff) * (1 - Vnext(x))));

			Vnext(n) = Vnext(n) + h2 * (P(k3) * ((Vnext(n) >= -P(Nmax) && Vnext(n) <= P(Nmax)) ? 1 - abs(Vnext(n) / P(Nmax)) : 0) * Vnext(i));

			Vnext(x) = Vnext(x) + h2 * (P(k2) * ((Vnext(x) >= 0 && Vnext(x) <= 1) ? 1 : 0) * Vnext(i));
			

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

		}
	}
	ifSIGNAL(P(signal), triangle)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{		
			Vnext(v) = P(Vdc) + P(Vamp) * ((4.0f * P(Vfreq) * (V(t) - P(Vdel)) - 2.0f * floorf((4.0f * P(Vfreq) * (V(t) - P(Vdel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Vfreq) * (V(t) - P(Vdel)) + 1.0f) / 2.0f)));
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			Vnext(x) = V(x) + H * (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * Vnext(i));

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			Vnext(n) = V(n) + H * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * Vnext(i));
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb vmp = P(Vdc) + P(Vamp) * ((4.0f * P(Vfreq) * (V(t) - P(Vdel)) - 2.0f * floorf((4.0f * P(Vfreq) * (V(t) - P(Vdel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Vfreq) * (V(t) - P(Vdel)) + 1.0f) / 2.0f)));
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			numb xmp = V(x) + H * 0.5f * (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * imp);

			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb nmp = V(n) + H * 0.5f * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * imp);
			numb tmp = V(t) + H * 0.5f;

			Vnext(v) = P(Vdc) + P(Vamp) * ((4.0f * P(Vfreq) * (tmp - P(Vdel)) - 2.0f * floorf((4.0f * P(Vfreq) * (tmp - P(Vdel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Vfreq) * (tmp - P(Vdel)) + 1.0f) / 2.0f)));
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			Vnext(x) = V(x) + H * (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * Vnext(i));

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			Vnext(n) = V(n) + H * (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * Vnext(i));
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb vmp = P(Vdc) + P(Vamp) * ((4.0f * P(Vfreq) * (V(t) - P(Vdel)) - 2.0f * floorf((4.0f * P(Vfreq) * (V(t) - P(Vdel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Vfreq) * (V(t) - P(Vdel)) + 1.0f) / 2.0f)));
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			numb kx1 = (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * imp);
			numb xmp = V(x) + H * 0.5f * kx1;

			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb kn1 = (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * Vnext(i));
			numb nmp = V(n) + H * 0.5f * kn1;
			numb tmp = V(t) + H * 0.5f;

			vmp = P(Vdc) + P(Vamp) * ((4.0f * P(Vfreq) * (tmp - P(Vdel)) - 2.0f * floorf((4.0f * P(Vfreq) * (tmp - P(Vdel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Vfreq) * (tmp - P(Vdel)) + 1.0f) / 2.0f)));
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			numb kx2 = (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * imp);
			xmp = V(x) + H * 0.5f * kx2;

			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb kn2 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * imp);
			nmp = V(n) + H * 0.5f * kn2;
			tmp = V(t) + H * 0.5f;

			vmp = P(Vdc) + P(Vamp) * ((4.0f * P(Vfreq) * (tmp - P(Vdel)) - 2.0f * floorf((4.0f * P(Vfreq) * (tmp - P(Vdel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Vfreq) * (tmp - P(Vdel)) + 1.0f) / 2.0f)));
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			numb kx3 = (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * imp);
			xmp = V(x) + H * kx3;

			if (xmp < 0) xmp = 0;
			if (xmp > 1) xmp = 1;

			numb kn3 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * imp);
			nmp = V(n) + H * kn3;
			tmp = V(t) + H;

			Vnext(v) = P(Vdc) + P(Vamp) * ((4.0f * P(Vfreq) * (tmp - P(Vdel)) - 2.0f * floorf((4.0f * P(Vfreq) * (tmp - P(Vdel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Vfreq) * (tmp - P(Vdel)) + 1.0f) / 2.0f)));
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * (1 - xmp)));
			numb kx4 = (P(k2) * ((xmp >= 0 && xmp <= 1) ? 1 : 0) * Vnext(i));
			Vnext(x) = V(x) + H * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6;

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			numb kn4 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? 1 - abs(nmp / P(Nmax)) : 0) * Vnext(i));
			Vnext(n) = V(n) + H * (kn1 + kn2 * 2 + kn3 * 2 + kn4) / 6;
			Vnext(t) = V(t) + H;

		}
		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = 0.5 * H - P(symmetry);
			numb h2 = 0.5 * H + P(symmetry);

			Vnext(v) = P(Vdc) + P(Vamp) * ((4.0f * P(Vfreq) * (V(t) - P(Vdel)) - 2.0f * floorf((4.0f * P(Vfreq) * (V(t) - P(Vdel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Vfreq) * (V(t) - P(Vdel)) + 1.0f) / 2.0f)));
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * (1 - V(x))));
			Vnext(x) = V(x) + h1 * (P(k2) * ((V(x) >= 0 && V(x) <= 1) ? 1 : 0) * Vnext(i));

			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

			Vnext(n) = V(n) + h1 * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? 1 - abs(V(n) / P(Nmax)) : 0) * Vnext(i));
			Vnext(t) = V(t) + h1;

			/////

			Vnext(t) = Vnext(t) + h2;
			Vnext(v) = P(Vdc) + P(Vamp) * sinf(2.0f * 3.141592f * P(Vfreq) * (Vnext(t) - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (Vnext(n) / (Vnext(x) + P(a))))) / ((P(Ron) * Vnext(x)) + (P(Roff) * (1 - Vnext(x))));

			Vnext(n) = P(Vdc) + P(Vamp) * ((4.0f * P(Vfreq) * (Vnext(t) - P(Vdel)) - 2.0f * floorf((4.0f * P(Vfreq) * (Vnext(t) - P(Vdel)) + 1.0f) / 2.0f)) * pow((-1), floorf((4.0f * P(Vfreq) * (Vnext(t) - P(Vdel)) + 1.0f) / 2.0f)));

			Vnext(x) = Vnext(x) + h2 * (P(k2) * ((Vnext(x) >= 0 && Vnext(x) <= 1) ? 1 : 0) * Vnext(i));


			if (Vnext(x) < 0) Vnext(x) = 0;
			if (Vnext(x) > 1) Vnext(x) = 1;

		}
		
	}
}
