#include "ostrovskii_modified.h"

#define name ostrovskii_modified

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
		Idc, Iamp, Ifreq, Idel, Idf, signal, 
        // Numerical integration
        method, 
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

__host__ __device__ __forceinline__ numb signal_square(numb t, const numb* parameters)
{
    const numb idc = P(Idc);
    const numb idel = P(Idel);
    const numb idf = P(Idf);
    const numb ifreq = P(Ifreq);
    const numb iamp = P(Iamp);

    const numb period = (numb)1.0 / ifreq;
    const numb duty = idf / ifreq;

    const numb tt = (t - idel) > (numb)0.0 ? (t - idel) : (duty + idel - t);
    const numb ph = fmod(tt, period);

    return idc + (ph < duty ? iamp : (numb)0.0);
}

__host__ __device__ __forceinline__ numb signal_sine(numb t, const numb* parameters)
{
    const numb pi2 = (numb)6.283185307179586476925286766559;

    return P(Idc) + P(Iamp) * sin(pi2 * P(Ifreq) * (t - P(Idel)));
}

__host__ __device__ __forceinline__ numb signal_triangle(numb t, const numb* parameters)
{
    numb q = (numb)4.0 * P(Ifreq) * (t - P(Idel));
    numb k = floor((q + (numb)1.0) / (numb)2.0);
    numb s = (((int)k) % 2 == 0) ? (numb)1.0 : (numb)-1.0;
    return P(Idc) + P(Iamp) * ((q - (numb)2.0 * k) * s);
}

__host__ __device__ __forceinline__ numb memristor_current(numb v, numb x, const numb* parameters)
{
    numb du = -v + P(Uvm);
    return (du > (numb)0.0)
        ? du * x / P(Ron_p) + P(Ilk)
        : du * x / P(Ron_n) - P(Ilk);
}

__host__ __device__ __forceinline__ numb diode_current(numb v, const numb* parameters)
{
    numb vu = v + P(Utd);
    numb invVt = (numb)1.0 / P(Vt);
    numb invVp = (numb)1.0 / P(Vp);

    return
        P(Is) * (exp(vu * invVt) - exp(-vu * invVt)) +
        (P(Ip) * invVp) * vu * exp(-(vu - P(Vp)) * invVp) +
        P(Iv) * (atan(P(D) * (vu - P(E))) + atan(P(D) * (vu + P(E))));
}

__host__ __device__ __forceinline__ numb x_rhs(numb v, numb x, const numb* parameters)
{
    const numb one = (numb)1.0;
    const numb xm1 = one - x;

    numb Xp__sigS = (P(Vh_p) - P(Vth_p)) * x + P(Vth_p);
    numb Xn__sigR = (P(Vh_n) - P(Vth_n)) * x + P(Vth_n);

    numb Vndr = (P(Uvm) - v - Xp__sigS) * (P(Uvm) - v - Xn__sigR);

    Xp__sigS = one / (one + exp(-Vndr / (P(Vs) * P(Vs))));
    Xn__sigR = one / (one + exp(-Vndr / (P(Vr) * P(Vr))));

    numb ax = P(A) * x;
    numb axm = P(A) * xm1;

    numb f_set =
        (one - exp(-(ax + P(Ds)))) * xm1 +
        x * (one - exp(-axm));

    numb f_reset =
        (one - exp(-ax)) * xm1 +
        x * (one - exp(-(axm + P(Dr))));

    return (Xp__sigS * f_set) / P(tau_s)
        - ((one - Xn__sigR) * f_reset) / P(tau_r);
}

template <typename InputSignal> __host__ __device__ __forceinline__ void rk4_scheme(numb* currentV, numb* nextV, numb* parameters, InputSignal signal_fn)
{
    const numb h = H;
    const numb h1 = (numb)0.5 * h;
    const numb h6 = h / (numb)6.0;

    numb v0 = V(v);
    numb x0 = V(x);
    numb t0 = V(t);

    numb invC = (numb)1.0 / P(C);

    numb sv, sx;
    numb vmp, xmp, tmp;
    numb kv, kx;

    // k1
    {
        const numb imp = signal_fn(t0, parameters);
        const numb Id = diode_current(v0, parameters);
        const numb Im = memristor_current(v0, x0, parameters);

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
        const numb imp = signal_fn(tmp, parameters);
        const numb Id = diode_current(vmp, parameters);
        const numb Im = memristor_current(vmp, xmp, parameters);

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
        const numb imp = signal_fn(tmp, parameters);
        const numb Id = diode_current(vmp, parameters);
        const numb Im = memristor_current(vmp, xmp, parameters);

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
        const numb imp = signal_fn(tmp, parameters);
        const numb Id = diode_current(vmp, parameters);
        const numb Im = memristor_current(vmp, xmp, parameters);

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