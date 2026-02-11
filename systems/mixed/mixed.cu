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
			Vnext(v) = P(Vdc) + (fmod((V(t) - P(Vdel)) > (numb)0.0 ? (V(t) - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - V(t)), (numb)1.0 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : -P(Vamp));
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			Vnext(x) = V(x) + H * (P(k2) * Vnext(i));

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			Vnext(n) = V(n) + H * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb vmp = P(Vdc) + (fmod((V(t) - P(Vdel)) > (numb)0.0 ? (V(t) - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - V(t)), (numb)1.0 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : -P(Vamp));
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			numb xmp = V(x) + H * (numb)0.5 * (P(k2) * imp);

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb nmp = V(n) + H * (numb)0.5 * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * imp);
			numb tmp = V(t) + H * (numb)0.5;

			Vnext(v) = P(Vdc) + (fmod((tmp - P(Vdel)) > (numb)0.0 ? (tmp - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - tmp), (numb)1.0 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : -P(Vamp));
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			Vnext(x) = V(x) + H * (P(k2) * Vnext(i));

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			Vnext(n) = V(n) + H * (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb vmp = P(Vdc) + (fmod((V(t) - P(Vdel)) > (numb)0.0 ? (V(t) - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - V(t)), (numb)1.0 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : -P(Vamp));
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			numb kx1 = (P(k2) * imp);
			numb xmp = V(x) + H * (numb)0.5 * kx1;

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb kn1 = (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * Vnext(i));
			numb nmp = V(n) + H * (numb)0.5 * kn1;
			numb tmp = V(t) + H * (numb)0.5;

			vmp = P(Vdc) + (fmod((tmp - P(Vdel)) > (numb)0.0 ? (tmp - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - tmp), (numb)1.0 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : -P(Vamp));
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			numb kx2 = (P(k2) * imp);
			xmp = V(x) + H * (numb)0.5 * kx2;

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb kn2 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * imp);
			nmp = V(n) + H * (numb)0.5 * kn2;
			tmp = V(t) + H * (numb)0.5;

			vmp = P(Vdc) + (fmod((tmp - P(Vdel)) > (numb)0.0 ? (tmp - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - tmp), (numb)1.0 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : -P(Vamp));
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			numb kx3 = (P(k2) * imp);
			xmp = V(x) + H * kx3;

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb kn3 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * imp);
			nmp = V(n) + H * kn3;
			tmp = V(t) + H;

			Vnext(v) = P(Vdc) + (fmod((tmp - P(Vdel)) > (numb)0.0 ? (tmp - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - tmp), (numb)1.0 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : -P(Vamp));
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			numb kx4 = (P(k2) * Vnext(i));
			Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			numb kn4 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(n) = V(n) + H * (kn1 + kn2 * (numb)2.0 + kn3 * (numb)2.0 + kn4) / (numb)6.0;
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), VariableSymmetryCD)
		{

			numb h1 = (numb)0.5 * H - P(symmetry);
			numb h2 = (numb)0.5 * H + P(symmetry);

			Vnext(v) = P(Vdc) + (fmod((V(t) - P(Vdel)) > (numb)0.0 ? (V(t) - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - V(t)), (numb)1.0 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : -P(Vamp));
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			Vnext(x) = V(x) + h1 * (P(k2) * Vnext(i));

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			Vnext(n) = V(n) + h1 * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(t) = V(t) + h1;

			/////

			Vnext(t) = Vnext(t) + h2;
			Vnext(v) = P(Vdc) + (fmod((Vnext(t) - P(Vdel)) > (numb)0.0 ? (Vnext(t) - P(Vdel)) : (P(Vdf) / P(Vfreq) + P(Vdel) - Vnext(t)), (numb)1.0 / P(Vfreq)) < P(Vdf) / P(Vfreq) ? P(Vamp) : -P(Vamp));
			Vnext(i) = (Vnext(v) - (P(k1) * (Vnext(n) / (Vnext(x) + P(a))))) / ((P(Ron) * Vnext(x)) + (P(Roff) * ((numb)1.0 - Vnext(x))));

			Vnext(n) = Vnext(n) + h2 * (P(k3) * ((Vnext(n) >= -P(Nmax) && Vnext(n) <= P(Nmax)) ? (numb)1.0 - abs(Vnext(n) / P(Nmax)) : (numb)0.0) * Vnext(i));

			Vnext(x) = Vnext(x) + h2 * (P(k2) * Vnext(i));


			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

		}
	}
	ifSIGNAL(P(signal), sine)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			Vnext(v) = P(Vdc) + P(Vamp) * sin((numb)2.0 * (numb)3.141592 * P(Vfreq) * (V(t) - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			Vnext(x) = V(x) + H * (P(k2) * Vnext(i));

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			Vnext(n) = V(n) + H * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb vmp = P(Vdc) + P(Vamp) * sin((numb)2.0 * (numb)3.141592 * P(Vfreq) * (V(t) - P(Vdel)));
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			numb xmp = V(x) + H * (numb)0.5 * (P(k2) * imp);

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb nmp = V(n) + H * (numb)0.5 * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * imp);
			numb tmp = V(t) + H * (numb)0.5;

			Vnext(v) = P(Vdc) + P(Vamp) * sin((numb)2.0 * (numb)3.141592 * P(Vfreq) * (tmp - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			Vnext(x) = V(x) + H * (P(k2) * Vnext(i));

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			Vnext(n) = V(n) + H * (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb vmp = P(Vdc) + P(Vamp) * sin((numb)2.0 * (numb)3.141592 * P(Vfreq) * (V(t) - P(Vdel)));
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			numb kx1 = (P(k2) * imp);
			numb xmp = V(x) + H * (numb)0.5 * kx1;

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb kn1 = (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * Vnext(i));
			numb nmp = V(n) + H * (numb)0.5 * kn1;
			numb tmp = V(t) + H * (numb)0.5;

			vmp = P(Vdc) + P(Vamp) * sin((numb)2.0 * (numb)3.141592 * P(Vfreq) * (tmp - P(Vdel)));
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			numb kx2 = (P(k2) * imp);
			xmp = V(x) + H * (numb)0.5 * kx2;

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb kn2 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * imp);
			nmp = V(n) + H * (numb)0.5 * kn2;
			tmp = V(t) + H * (numb)0.5;

			vmp = P(Vdc) + P(Vamp) * sin((numb)2.0 * (numb)3.141592 * P(Vfreq) * (tmp - P(Vdel)));
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			numb kx3 = (P(k2) * imp);
			xmp = V(x) + H * kx3;

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb kn3 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * imp);
			nmp = V(n) + H * kn3;
			tmp = V(t) + H;

			Vnext(v) = P(Vdc) + P(Vamp) * sin((numb)2.0 * (numb)3.141592 * P(Vfreq) * (tmp - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			numb kx4 = (P(k2) * Vnext(i));
			Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			numb kn4 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(n) = V(n) + H * (kn1 + kn2 * (numb)2.0 + kn3 * (numb)2.0 + kn4) / (numb)6.0;
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), VariableSymmetryCD)
		{

			numb h1 = (numb)0.5 * H - P(symmetry);
			numb h2 = (numb)0.5 * H + P(symmetry);

			Vnext(v) = P(Vdc) + P(Vamp) * sin((numb)2.0 * (numb)3.141592 * P(Vfreq) * (V(t) - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			Vnext(x) = V(x) + h1 * (P(k2) * Vnext(i));

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			Vnext(n) = V(n) + h1 * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(t) = V(t) + h1;

			/////

			Vnext(t) = Vnext(t) + h2;
			Vnext(v) = P(Vdc) + P(Vamp) * sin((numb)2.0 * (numb)3.141592 * P(Vfreq) * (Vnext(t) - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (Vnext(n) / (Vnext(x) + P(a))))) / ((P(Ron) * Vnext(x)) + (P(Roff) * ((numb)1.0 - Vnext(x))));

			Vnext(n) = Vnext(n) + h2 * (P(k3) * ((Vnext(n) >= -P(Nmax) && Vnext(n) <= P(Nmax)) ? (numb)1.0 - abs(Vnext(n) / P(Nmax)) : (numb)0.0) * Vnext(i));

			Vnext(x) = Vnext(x) + h2 * (P(k2) * Vnext(i));


			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

		}
	}
	ifSIGNAL(P(signal), triangle)
	{
		ifMETHOD(P(method), ExplicitEuler)
		{
			Vnext(v) = P(Vdc) + P(Vamp) * (((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) - (numb)2.0 * floor((((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) + (numb)1.0) / (numb)2.0))));
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			Vnext(x) = V(x) + H * (P(k2) * Vnext(i));

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			Vnext(n) = V(n) + H * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), ExplicitMidpoint)
		{
			numb vmp = P(Vdc) + P(Vamp) * (((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) - (numb)2.0 * floor((((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) + (numb)1.0) / (numb)2.0))));
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			numb xmp = V(x) + H * (numb)0.5 * (P(k2) * imp);

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb nmp = V(n) + H * (numb)0.5 * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * imp);
			numb tmp = V(t) + H * (numb)0.5;

			Vnext(v) = P(Vdc) + P(Vamp) * (((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) - (numb)2.0 * floor((((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) + (numb)1.0) / (numb)2.0))));
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			Vnext(x) = V(x) + H * (P(k2) * Vnext(i));

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			Vnext(n) = V(n) + H * (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(t) = V(t) + H;
		}
		ifMETHOD(P(method), ExplicitRungeKutta4)
		{
			numb vmp = P(Vdc) + P(Vamp) * (((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) - (numb)2.0 * floor((((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) + (numb)1.0) / (numb)2.0))));
			numb imp = (vmp - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			numb kx1 = (P(k2) * imp);
			numb xmp = V(x) + H * (numb)0.5 * kx1;

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb kn1 = (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * Vnext(i));
			numb nmp = V(n) + H * (numb)0.5 * kn1;
			numb tmp = V(t) + H * (numb)0.5;

			vmp = P(Vdc) + P(Vamp) * (((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) - (numb)2.0 * floor((((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) + (numb)1.0) / (numb)2.0))));
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			numb kx2 = (P(k2) * imp);
			xmp = V(x) + H * (numb)0.5 * kx2;

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb kn2 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * imp);
			nmp = V(n) + H * (numb)0.5 * kn2;
			tmp = V(t) + H * (numb)0.5;

			vmp = P(Vdc) + P(Vamp) * (((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) - (numb)2.0 * floor((((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) + (numb)1.0) / (numb)2.0))));
			imp = (vmp - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			numb kx3 = (P(k2) * imp);
			xmp = V(x) + H * kx3;

			if (xmp < (numb)0.0) xmp = (numb)0.0;
			if (xmp > (numb)1.0) xmp = (numb)1.0;

			numb kn3 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * imp);
			nmp = V(n) + H * kn3;
			tmp = V(t) + H;

			Vnext(v) = P(Vdc) + P(Vamp) * (((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) - (numb)2.0 * floor((((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Vfreq) * (tmp - P(Vdel)) + (numb)1.0) / (numb)2.0))));
			Vnext(i) = (Vnext(v) - (P(k1) * (nmp / (xmp + P(a))))) / ((P(Ron) * xmp) + (P(Roff) * ((numb)1.0 - xmp)));
			numb kx4 = (P(k2) * Vnext(i));
			Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			numb kn4 = (P(k3) * ((nmp >= -P(Nmax) && nmp <= P(Nmax)) ? (numb)1.0 - abs(nmp / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(n) = V(n) + H * (kn1 + kn2 * (numb)2.0 + kn3 * (numb)2.0 + kn4) / (numb)6.0;
			Vnext(t) = V(t) + H;

		}
		ifMETHOD(P(method), VariableSymmetryCD)
		{
			numb h1 = (numb)0.5 * H - P(symmetry);
			numb h2 = (numb)0.5 * H + P(symmetry);

			Vnext(v) = P(Vdc) + P(Vamp) * (((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) - (numb)2.0 * floor((((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Vfreq) * (V(t) - P(Vdel)) + (numb)1.0) / (numb)2.0))));
			Vnext(i) = (Vnext(v) - (P(k1) * (V(n) / (V(x) + P(a))))) / ((P(Ron) * V(x)) + (P(Roff) * ((numb)1.0 - V(x))));
			Vnext(x) = V(x) + h1 * (P(k2) * Vnext(i));

			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

			Vnext(n) = V(n) + h1 * (P(k3) * ((V(n) >= -P(Nmax) && V(n) <= P(Nmax)) ? (numb)1.0 - abs(V(n) / P(Nmax)) : (numb)0.0) * Vnext(i));
			Vnext(t) = V(t) + h1;

			/////

			Vnext(t) = Vnext(t) + h2;
			Vnext(v) = P(Vdc) + P(Vamp) * sin((numb)2.0 * (numb)3.141592 * P(Vfreq) * (Vnext(t) - P(Vdel)));
			Vnext(i) = (Vnext(v) - (P(k1) * (Vnext(n) / (Vnext(x) + P(a))))) / ((P(Ron) * Vnext(x)) + (P(Roff) * ((numb)1.0 - Vnext(x))));

			Vnext(n) = P(Vdc) + P(Vamp) * (((numb)4.0 * P(Vfreq) * (Vnext(t) - P(Vdel)) - (numb)2.0 * floor((((numb)4.0 * P(Vfreq) * (Vnext(t) - P(Vdel)) + (numb)1.0) / (numb)2.0))) * pow((numb)-1.0, floor((((numb)4.0 * P(Vfreq) * (Vnext(t) - P(Vdel)) + (numb)1.0) / (numb)2.0))));

			Vnext(x) = Vnext(x) + h2 * (P(k2) * Vnext(i));


			if (Vnext(x) < (numb)0.0) Vnext(x) = (numb)0.0;
			if (Vnext(x) > (numb)1.0) Vnext(x) = (numb)1.0;

		}

	}
}
