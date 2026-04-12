#include "khanov.h"

#define name khanov

namespace attributes
{
    enum variables { v, x, i, t};
    enum parameters { 
		// Circuit
		Utd, Uvm, C,
		// Tunnel diode GI403
		Is, Vt, Vp, Ip, Iv, D, E,
		// Volatile memristor AND_TS
		Ron_p, Ron_n, Vth_p, Vh_p, Vth_n, Vh_n, tau_s, tau_r, Vs, Vr, A, Ds, Dr, Ilk, 
		// Input signal
		Idc, Iamp, Ifreq, Idel, Idf, Ispc, Inum, Iinc, signal, 
        // Numerical integration
        method, 
		COUNT 
	};
    enum waveforms { square, square_multi, sine, triangle };
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

__host__ __device__ __forceinline__ numb signal_square(numb t, numb* parameters)
{
    numb idel = P(Idel);
    numb ifreq = P(Ifreq);

    numb period = (numb)1.0 / ifreq; // Period in seconds
    numb duty = P(Idf) / ifreq; // Signal "on" time in seconds

    numb tt = (t - idel) > (numb)0.0 ? (t - idel) : (duty + idel - t);
    numb ph = fmod(tt, period);

    return P(Idc) + (ph < duty ? P(Iamp) : (numb)0.0);
}

__host__ __device__ __forceinline__ numb signal_square_multi(numb t, numb* parameters)
{
    numb idel = P(Idel);
    numb ifreq = P(Ifreq);

    numb period = (numb)1.0 / ifreq; // Period in seconds
    numb duty = P(Idf) / ifreq; // Signal "on" time in seconds
    numb space = P(Ispc) / ifreq; // Space between "on"s in seconds

    int num = (int)P(Inum);
    numb Iac = (numb)0.0;
    numb shift, ph;
    for (int i = 0; i < num; i++)
    {
        shift = space * i;
        ph = fmod((t - idel - shift) > (numb)0.0 ? (t - idel - shift) : (duty + idel + shift - t), period);
        if (ph < duty) Iac += P(Iamp) * ((numb)1.0 + P(Iinc) * i);
    }

    return P(Idc) + Iac;
}

__host__ __device__ __forceinline__ numb signal_sine(numb t, numb* parameters)
{
    numb pi2 = (numb)6.283185307179586476925286766559;

    return P(Idc) + P(Iamp) * sin(pi2 * P(Ifreq) * (t - P(Idel)));
}

__host__ __device__ __forceinline__ numb signal_triangle(numb t, numb* parameters)
{
    numb q = (numb)4.0 * P(Ifreq) * (t - P(Idel));
    numb k = floor((q + (numb)1.0) / (numb)2.0);
    numb s = (((int)k) % 2 == 0) ? (numb)1.0 : (numb)-1.0;
    return P(Idc) + P(Iamp) * ((q - (numb)2.0 * k) * s);
}

__host__ __device__ __forceinline__ numb memristor_current(numb v, numb x, numb* parameters)
{
    numb du = -v + P(Uvm);
    return (du > (numb)0.0)
        ? du * x / P(Ron_p) + P(Ilk)
        : du * x / P(Ron_n) - P(Ilk);
}

__host__ __device__ __forceinline__ numb diode_current(numb v, numb* parameters)
{
    numb vu = v + P(Utd);
    numb invVt = (numb)1.0 / P(Vt);
    numb invVp = (numb)1.0 / P(Vp);

    return
        P(Is) * (exp(vu * invVt) - exp(-vu * invVt)) +
        (P(Ip) * invVp) * vu * exp(-(vu - P(Vp)) * invVp) +
        P(Iv) * (atan(P(D) * (vu - P(E))) + atan(P(D) * (vu + P(E))));
}

__host__ __device__ __forceinline__ numb x_rhs(numb v, numb x, numb* parameters)
{
    numb one = (numb)1.0;
    numb xm1 = one - x;

    numb Xp = (P(Vh_p) - P(Vth_p)) * x + P(Vth_p);
    numb Xn = (P(Vh_n) - P(Vth_n)) * x + P(Vth_n);

    numb Vndr = (P(Uvm) - v - Xp) * (P(Uvm) - v - Xn);

    numb sigS = one / (one + exp(-Vndr / (P(Vs) * P(Vs))));
    numb sigR = one / (one + exp(-Vndr / (P(Vr) * P(Vr))));

    numb ax = P(A) * x;
    numb axm = P(A) * xm1;

    numb f_set =
        (one - exp(-(ax + P(Ds)))) * xm1 +
        x * (one - exp(-axm));

    numb f_reset =
        (one - exp(-ax)) * xm1 +
        x * (one - exp(-(axm + P(Dr))));

    return (sigS * f_set) / P(tau_s)
        - ((one - sigR) * f_reset) / P(tau_r);
}

template <typename InputSignal> __host__ __device__ __forceinline__ void rk4_scheme(numb* currentV, numb* nextV, numb* parameters, InputSignal signal_fn)
{
    numb h = H;
    numb h1 = (numb)0.5 * h;
    numb h6 = h / (numb)6.0;

    numb v0 = V(v);
    numb x0 = V(x);
    numb t0 = V(t);

    numb invC = (numb)1.0 / P(C);

    numb sv, sx;
    numb vmp, xmp, tmp;
    numb kv, kx;

    // k1
    {
        numb imp = signal_fn(t0, parameters);
        numb Id = diode_current(v0, parameters);
        numb Im = memristor_current(v0, x0, parameters);

        kv = (imp + Im - Id) * invC;
        kx = x_rhs(v0, x0, parameters);

        sv = kv;
        sx = kx;

        vmp = v0 + h1 * kv;
        xmp = x0 + h1 * kx;
        tmp = t0 + h1;
    }
    // k2
    {
        numb imp = signal_fn(tmp, parameters);
        numb Id = diode_current(vmp, parameters);
        numb Im = memristor_current(vmp, xmp, parameters);

        kv = (imp + Im - Id) * invC;
        kx = x_rhs(vmp, xmp, parameters);

        sv += (numb)2.0 * kv;
        sx += (numb)2.0 * kx;

        vmp = v0 + h1 * kv;
        xmp = x0 + h1 * kx;
        tmp = t0 + h1;
    }
    // k3
    {
        numb imp = signal_fn(tmp, parameters);
        numb Id = diode_current(vmp, parameters);
        numb Im = memristor_current(vmp, xmp, parameters);

        kv = (imp + Im - Id) * invC;
        kx = x_rhs(vmp, xmp, parameters);

        sv += (numb)2.0 * kv;
        sx += (numb)2.0 * kx;

        vmp = v0 + h * kv;
        xmp = x0 + h * kx;
        tmp = t0 + h;
    }
    // k4
    {
        numb imp = signal_fn(tmp, parameters);
        numb Id = diode_current(vmp, parameters);
        numb Im = memristor_current(vmp, xmp, parameters);

        kv = (imp + Im - Id) * invC;
        kx = x_rhs(vmp, xmp, parameters);

        sv += kv;
        sx += kx;

        Vnext(i) = imp;
    }

    Vnext(v) = v0 + h6 * sv;
    Vnext(x) = x0 + h6 * sx;
    Vnext(t) = t0 + h;
}

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters, PerThread* pt)
{
    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        ifSIGNAL(P(signal), square)
        {
            rk4_scheme(currentV, nextV, parameters, signal_square);
        }

        ifSIGNAL(P(signal), square_multi)
        {
            rk4_scheme(currentV, nextV, parameters, signal_square_multi);
        }

        ifSIGNAL(P(signal), sine)
        {
            rk4_scheme(currentV, nextV, parameters, signal_sine);
        }

        ifSIGNAL(P(signal), triangle)
        {
            rk4_scheme(currentV, nextV, parameters, signal_triangle);
        }
    }
}