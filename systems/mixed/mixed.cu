#include "main.h"
#include "mixed.h"

namespace attributes
{
    enum variables { v, I, x, N, t };
    enum parameters { k1, k2, k3, a, Nmax, Ron, Roff, A, nu, T, method, signal, COUNT };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_mixed(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_mixed);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_mixed(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_mixed, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_mixed, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBSCAN_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4),
            H);
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_mixed, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__  void finiteDifferenceScheme_mixed(numb* currentV, numb* nextV, numb* parameters)
{
    // V - input variable
    // x - state variable
    // N - state variable

    // k1, k2, k3 - parameters
    // a, Nmax - parameters
    // Ron, Roff - parameters

    // V(t) = R(x)*I(t) + deltaV(x, N)
    // I(t) = ( V(t) - deltaV(x, N) ) / R(x)
    // dx/dt = k2*fw(x)*I(t)
    // dN/dt = k3*fv(N)*I(t)
    // deltaV(x, N) = k1 * (N / (x+a))
    // fv(N) = 1 - |N / Nmax|, N e [-Nmax, Nmax] or = 0, N !e [-Nmax, Nmax]
    // k3 = k3+, N >= 0 or = k3-, N < 0
    // fw(x) = 1, x e [0, 1] or = 0, x !e [0, 1]
    // R = Ron*x + Roff*(1-x)

    // V(t) = A*sin(2 * pi * nu * t)

    ifMETHOD(P(method), ExplicitEuler)
    {
        numb deltav = P(k1) * (V(N) / (V(x) + P(a)));
        numb R = (P(Ron) * V(x)) + (P(Roff) * (1 - V(x)));
        numb I = (V(v) - deltav) / R;

        numb fw = (V(x) >= 0 && V(x) <= 1) ? 1 : 0;
        numb fv = (V(N) >= -P(Nmax) && V(N) <= P(Nmax)) ? 1 - abs(V(N) / P(Nmax)) : 0;

        Vnext(v) = P(A) * sin(2 * 3.141592 * P(nu) * V(t));
        Vnext(I) = I;
        Vnext(x) = V(x) + H * (P(k2) * fw * I);
        if (Vnext(x) < 0) Vnext(x) = 0;
        if (Vnext(x) > 1) Vnext(x) = 1;
        Vnext(N) = V(N) + H * (P(k3) * fv * I);
        Vnext(t) = V(t) + H;
    }

    /*ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + H * 0.5 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + P(Iext));
        numb ymp = V(y) + H * 0.5 * (P(c) - P(d) * V(x) * V(x) - V(y));
        numb zmp = V(z) + H * 0.5 * (P(r) * (P(s) * (V(x) + P(e)) - V(z)));

        Vnext(x) = V(x) + H * (ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext));
        Vnext(y) = V(y) + H * (P(c) - P(d) * xmp * xmp - ymp);
        Vnext(z) = V(z) + H * (P(r) * (P(s) * (xmp + P(e)) - zmp));
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + P(Iext);
        numb ky1 = P(c) - P(d) * V(x) * V(x) - V(y);
        numb kz1 = P(r) * (P(s) * (V(x) + P(e)) - V(z));

        numb xmp = V(x) + 0.5 * H * kx1;
        numb ymp = V(y) + 0.5 * H * ky1;
        numb zmp = V(z) + 0.5 * H * kz1;

        numb kx2 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext);
        numb ky2 = P(c) - P(d) * xmp * xmp - ymp;
        numb kz2 = P(r) * (P(s) * (xmp + P(e)) - zmp);

        xmp = V(x) + 0.5 * H * kx2;
        ymp = V(y) + 0.5 * H * ky2;
        zmp = V(z) + 0.5 * H * kz2;

        numb kx3 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext);
        numb ky3 = P(c) - P(d) * xmp * xmp - ymp;
        numb kz3 = P(r) * (P(s) * (xmp + P(e)) - zmp);

        xmp = V(x) + H * kx3;
        ymp = V(y) + H * ky3;
        zmp = V(z) + H * kz3;

        numb kx4 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + P(Iext);
        numb ky4 = P(c) - P(d) * xmp * xmp - ymp;
        numb kz4 = P(r) * (P(s) * (xmp + P(e)) - zmp);

        Vnext(x) = V(x) + H * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4) / 6.0;
        Vnext(y) = V(y) + H * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4) / 6.0;
        Vnext(z) = V(z) + H * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4) / 6.0;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5 * H - P(symmetry);
        numb h2 = 0.5 * H + P(symmetry);

        numb xmp = V(x) + h1 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + P(Iext));
        numb ymp = V(y) + h1 * (P(c) - P(d) * xmp * xmp - V(y));
        numb zmp = V(z) + h1 * (P(r) * (P(s) * (xmp + P(e)) - V(z)));

        Vnext(z) = (zmp + P(r) * P(s) * (xmp + P(e)) * h2) / (1 + P(r) * h2);
        Vnext(y) = (ymp + (P(c) - P(d) * xmp * xmp) * h2) / (1 + h2);

        Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - Vnext(z) + P(Iext));
        Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * Vnext(x) * Vnext(x) * Vnext(x) + P(b) * Vnext(x) * Vnext(x) - Vnext(z) + P(Iext));
    }*/
}
