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

template <typename T> __host__ __device__ __forceinline__ T signal_square(T t, const numb* parameters)
{
    const T idc = P(Idc);
    const T idel = P(Idel);
    const T idf = P(Idf);
    const T ifreq = P(Ifreq);
    const T iamp = P(Iamp);

    const T period = (T)1.0 / ifreq;
    const T duty = idf / ifreq;

    const T tt = (t - idel) > (T)0.0 ? (t - idel) : (duty + idel - t);
    const T ph = fmod(tt, period);

    return idc + (ph < duty ? iamp : (T)0.0);
}

template <typename T> __host__ __device__ __forceinline__ T signal_sine(T t, const numb* parameters)
{
    const T pi2 = (T)6.283185307179586476925286766559;
    return P(Idc) + P(Iamp) * sin(pi2 * P(Ifreq) * (t - P(Idel)));
}

template <typename T> __host__ __device__ __forceinline__ T signal_triangle(T t, const numb* parameters)
{
    const T q = (T)4.0 * P(Ifreq) * (t - P(Idel));
    const T k = floor((q + (T)1.0) / (T)2.0);
    const T s = (((int)k) % 2 == 0) ? (T)1.0 : (T)-1.0;
    return P(Idc) + P(Iamp) * ((q - (T)2.0 * k) * s);
}

template <typename T> __host__ __device__ __forceinline__ T memristor_current(T v, T x, const numb* parameters)
{
    const T du = -v + P(Uvm);
    return (du > (T)0.0)
        ? du * x / P(Ron_p) + P(Ilk)
        : du * x / P(Ron_n) - P(Ilk);
}

template <typename T> __host__ __device__ __forceinline__ T diode_current(T v, const numb* parameters)
{
    const T vu = v + P(Utd);
    const T invVt = (T)1.0 / P(Vt);
    const T invVp = (T)1.0 / P(Vp);

    return
        P(Is) * (exp(vu * invVt) - exp(-vu * invVt)) +
        (P(Ip) * invVp) * vu * exp(-(vu - P(Vp)) * invVp) +
        P(Iv) * (atan(P(D) * (vu - P(E))) + atan(P(D) * (vu + P(E))));
}

template <typename T> __host__ __device__ __forceinline__ T x_rhs(T v, T x, const numb* parameters)
{
    const T one = (T)1.0;
    const T xm1 = one - x;
    const T u = -v + P(Uvm);

    const T xp = (P(Vh_p) - P(Vth_p)) * x + P(Vth_p);
    const T xn = (P(Vh_n) - P(Vth_n)) * x + P(Vth_n);

    const T dup = u - xp;
    const T dun = u - xn;

    const T sig_s = one / (one + exp(-(dup * dun) / (P(Vs) * P(Vs))));
    const T sig_r = one / (one + exp(-(dup * dun) / (P(Vr) * P(Vr))));

    const T ax = P(A) * x;
    const T axm = P(A) * xm1;

    const T f_set =
        (one - exp(-(ax + P(Ds)))) * xm1 +
        x * (one - exp(-axm));

    const T f_reset =
        (one - exp(-ax)) * xm1 +
        x * (one - exp(-(axm + P(Dr))));

    return (sig_s * f_set) / P(tau_s)
        - ((one - sig_r) * f_reset) / P(tau_r);
}

template <typename TSignal> __host__ __device__ __forceinline__ void rk4_scheme(numb* currentV, numb* nextV, numb* parameters, TSignal signal_fn)
{
    const numb v0 = V(v);
    const numb x0 = V(x);
    const numb t0 = V(t);

    const numb h = H;
    const numb h1 = (numb)0.5 * h;
    const numb h6 = h / (numb)6.0;
    const numb invC = (numb)1.0 / P(C);

    numb sv, sx;
    numb vmp, xmp, tmp;
    numb kv, kx;

    // k1
    {
        const numb imp = signal_fn(t0, parameters);
        const numb Id = diode_current<numb>(v0, parameters);
        const numb Im = memristor_current<numb>(v0, x0, parameters);

        kv = (imp + Im - Id) * invC;
        kx = x_rhs<numb>(v0, x0, parameters);

        sv = kv;
        sx = kx;

        vmp = v0 + h1 * kv;
        xmp = x0 + h1 * kx;
        tmp = t0 + h1;
    }
    // k2
    {
        const numb imp = signal_fn(tmp, parameters);
        const numb Id = diode_current<numb>(vmp, parameters);
        const numb Im = memristor_current<numb>(vmp, xmp, parameters);

        kv = (imp + Im - Id) * invC;
        kx = x_rhs<numb>(vmp, xmp, parameters);

        sv += (numb)2.0 * kv;
        sx += (numb)2.0 * kx;

        vmp = v0 + h1 * kv;
        xmp = x0 + h1 * kx;
        tmp = t0 + h1;
    }
    // k3
    {
        const numb imp = signal_fn(tmp, parameters);
        const numb Id = diode_current<numb>(vmp, parameters);
        const numb Im = memristor_current<numb>(vmp, xmp, parameters);

        kv = (imp + Im - Id) * invC;
        kx = x_rhs<numb>(vmp, xmp, parameters);

        sv += (numb)2.0 * kv;
        sx += (numb)2.0 * kx;

        vmp = v0 + h * kv;
        xmp = x0 + h * kx;
        tmp = t0 + h;
    }
    // k4
    {
        const numb imp = signal_fn(tmp, parameters);
        const numb Id = diode_current<numb>(vmp, parameters);
        const numb Im = memristor_current<numb>(vmp, xmp, parameters);

        kv = (imp + Im - Id) * invC;
        kx = x_rhs<numb>(vmp, xmp, parameters);

        sv += kv;
        sx += kx;

        Vnext(i) = imp;
    }

    Vnext(v) = v0 + h6 * sv;
    Vnext(x) = x0 + h6 * sx;
    Vnext(t) = t0 + h;
}

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters, Computation* data)
{
    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        ifSIGNAL(P(signal), square)
        {
            rk4_scheme(currentV, nextV, parameters, signal_square<numb>);
        }

        ifSIGNAL(P(signal), sine)
        {
            rk4_scheme(currentV, nextV, parameters, signal_sine<numb>);
        }

        ifSIGNAL(P(signal), triangle)
        {
            rk4_scheme(currentV, nextV, parameters, signal_triangle<numb>);
        }
    }
}