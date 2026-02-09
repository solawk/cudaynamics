#include "hindmarsh_rose.h"

#define name hindmarsh_rose

namespace attributes
{
    enum variables { x, y, z, i, t };
    enum parameters { a, b, c, d, r, s, e, Idc, Iamp, Ifreq, Idel, Idf, symmetry, signal, method, COUNT };
    enum waveforms { square, sine, triangle };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD};
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
            Vnext(i) = P(Idc) + (std::fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(t) = V(t) + H;
            Vnext(x) = V(x) + H * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + Vnext(i));
            Vnext(y) = V(y) + H * (P(c) - P(d) * V(x) * V(x) - V(y));
            Vnext(z) = V(z) + H * (P(r) * (P(s) * (V(x) + P(e)) - V(z)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + (std::fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb tmp = V(t) + H * (numb)0.5;
            numb xmp = V(x) + H * (numb)0.5 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + imp);
            numb ymp = V(y) + H * (numb)0.5 * (P(c) - P(d) * V(x) * V(x) - V(y));
            numb zmp = V(z) + H * (numb)0.5 * (P(r) * (P(s) * (V(x) + P(e)) - V(z)));

            Vnext(i) = P(Idc) + (std::fmod((tmp - P(Idel)) > (numb)0 ? (tmp - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - tmp), (numb)1.0 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            Vnext(t) = V(t) + H;
            Vnext(x) = V(x) + H * (ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + Vnext(i));
            Vnext(y) = V(y) + H * (P(c) - P(d) * xmp * xmp - ymp);
            Vnext(z) = V(z) + H * (P(r) * (P(s) * (xmp + P(e)) - zmp));
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb i1 = P(Idc) + (std::fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb kt1 = V(t) + (numb)0.5 * H;

            numb kx1 = V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + i1;
            numb ky1 = P(c) - P(d) * V(x) * V(x) - V(y);
            numb kz1 = P(r) * (P(s) * (V(x) + P(e)) - V(z));

            numb xmp = V(x) + (numb)0.5 * H * kx1;
            numb ymp = V(y) + (numb)0.5 * H * ky1;
            numb zmp = V(z) + (numb)0.5 * H * kz1;


            numb i2 = P(Idc) + (std::fmod((kt1 - P(Idel)) > (numb)0 ? (kt1 - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - kt1), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb kx2 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + i2;
            numb ky2 = P(c) - P(d) * xmp * xmp - ymp;
            numb kz2 = P(r) * (P(s) * (xmp + P(e)) - zmp);

            xmp = V(x) + (numb)0.5 * H * kx2;
            ymp = V(y) + (numb)0.5 * H * ky2;
            zmp = V(z) + (numb)0.5 * H * kz2;

            numb kx3 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + i2;
            numb ky3 = P(c) - P(d) * xmp * xmp - ymp;
            numb kz3 = P(r) * (P(s) * (xmp + P(e)) - zmp);
            Vnext(t) = V(t) + H;

            xmp = V(x) + H * kx3;
            ymp = V(y) + H * ky3;
            zmp = V(z) + H * kz3;

            numb i3 = P(Idc) + (std::fmod((Vnext(t) - P(Idel)) > (numb)0 ? (Vnext(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - Vnext(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb kx4 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + i3;
            numb ky4 = P(c) - P(d) * xmp * xmp - ymp;
            numb kz4 = P(r) * (P(s) * (xmp + P(e)) - zmp);

            Vnext(i) = i3;
            Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
            Vnext(y) = V(y) + H * (ky1 + (numb)2.0 * ky2 + (numb)2.0 * ky3 + ky4) / (numb)6.0;
            Vnext(z) = V(z) + H * (kz1 + (numb)2.0 * kz2 + (numb)2.0 * kz3 + kz4) / (numb)6.0;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = (numb)0.5 * H - P(symmetry);
            numb h2 = (numb)0.5 * H + P(symmetry);


            Vnext(i) = P(Idc) + (std::fmod((V(t) - P(Idel)) > (numb)0 ? (V(t) - P(Idel)) : (P(Idf) / P(Ifreq) + P(Idel) - V(t)), (numb)1 / P(Ifreq)) < P(Idf) / P(Ifreq) ? P(Iamp) : (numb)0.0);
            numb xmp = V(x) + h1 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + Vnext(i));
            numb ymp = V(y) + h1 * (P(c) - P(d) * xmp * xmp - V(y));
            numb zmp = V(z) + h1 * (P(r) * (P(s) * (xmp + P(e)) - V(z)));

            Vnext(z) = (zmp + P(r) * P(s) * (xmp + P(e)) * h2) / ((numb)1 + P(r) * h2);
            Vnext(y) = (ymp + (P(c) - P(d) * xmp * xmp) * h2) / ((numb)1 + h2);

            Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - Vnext(z) + Vnext(i));
            Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * Vnext(x) * Vnext(x) * Vnext(x) + P(b) * Vnext(x) * Vnext(x) - Vnext(z) + Vnext(i));
            Vnext(t) = V(t) + H;
        }
    }
    ifSIGNAL(P(signal), sine)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * std::sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            Vnext(t) = V(t) + H;
            Vnext(x) = V(x) + H * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + Vnext(i));
            Vnext(y) = V(y) + H * (P(c) - P(d) * V(x) * V(x) - V(y));
            Vnext(z) = V(z) + H * (P(r) * (P(s) * (V(x) + P(e)) - V(z)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + P(Iamp) * std::sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            numb tmp = V(t) + H * (numb)0.5;
            numb xmp = V(x) + H * (numb)0.5 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + imp);
            numb ymp = V(y) + H * (numb)0.5 * (P(c) - P(d) * V(x) * V(x) - V(y));
            numb zmp = V(z) + H * (numb)0.5 * (P(r) * (P(s) * (V(x) + P(e)) - V(z)));

            Vnext(i) = P(Idc) + P(Iamp) * std::sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (tmp - P(Idel)));
            Vnext(t) = V(t) + H;
            Vnext(x) = V(x) + H * (ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + Vnext(i));
            Vnext(y) = V(y) + H * (P(c) - P(d) * xmp * xmp - ymp);
            Vnext(z) = V(z) + H * (P(r) * (P(s) * (xmp + P(e)) - zmp));
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb i1 = P(Idc) + P(Iamp) * std::sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            numb kt1 = V(t) + (numb)0.5 * H;

            numb kx1 = V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + i1;
            numb ky1 = P(c) - P(d) * V(x) * V(x) - V(y);
            numb kz1 = P(r) * (P(s) * (V(x) + P(e)) - V(z));

            numb xmp = V(x) + (numb)0.5 * H * kx1;
            numb ymp = V(y) + (numb)0.5 * H * ky1;
            numb zmp = V(z) + (numb)0.5 * H * kz1;


            numb i2 = P(Idc) + P(Iamp) * std::sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (kt1 - P(Idel)));
            numb kx2 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + i2;
            numb ky2 = P(c) - P(d) * xmp * xmp - ymp;
            numb kz2 = P(r) * (P(s) * (xmp + P(e)) - zmp);

            xmp = V(x) + (numb)0.5 * H * kx2;
            ymp = V(y) + (numb)0.5 * H * ky2;
            zmp = V(z) + (numb)0.5 * H * kz2;

            numb kx3 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + i2;
            numb ky3 = P(c) - P(d) * xmp * xmp - ymp;
            numb kz3 = P(r) * (P(s) * (xmp + P(e)) - zmp);
            Vnext(t) = V(t) + H;

            xmp = V(x) + H * kx3;
            ymp = V(y) + H * ky3;
            zmp = V(z) + H * kz3;

            numb i3 = P(Idc) + P(Iamp) * std::sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (Vnext(t) - P(Idel)));
            numb kx4 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + i3;
            numb ky4 = P(c) - P(d) * xmp * xmp - ymp;
            numb kz4 = P(r) * (P(s) * (xmp + P(e)) - zmp);

            Vnext(i) = i3;
            Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
            Vnext(y) = V(y) + H * (ky1 + (numb)2.0 * ky2 + (numb)2.0 * ky3 + ky4) / (numb)6.0;
            Vnext(z) = V(z) + H * (kz1 + (numb)2.0 * kz2 + (numb)2.0 * kz3 + kz4) / (numb)6.0;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = (numb)0.5 * H - P(symmetry);
            numb h2 = (numb)0.5 * H + P(symmetry);


            Vnext(i) = P(Idc) + P(Iamp) * std::sin((numb)2.0 * (numb)3.141592 * P(Ifreq) * (V(t) - P(Idel)));
            numb xmp = V(x) + h1 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + Vnext(i));
            numb ymp = V(y) + h1 * (P(c) - P(d) * xmp * xmp - V(y));
            numb zmp = V(z) + h1 * (P(r) * (P(s) * (xmp + P(e)) - V(z)));

            Vnext(z) = (zmp + P(r) * P(s) * (xmp + P(e)) * h2) / (1 + P(r) * h2);
            Vnext(y) = (ymp + (P(c) - P(d) * xmp * xmp) * h2) / (1 + h2);

            Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - Vnext(z) + Vnext(i));
            Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * Vnext(x) * Vnext(x) * Vnext(x) + P(b) * Vnext(x) * Vnext(x) - Vnext(z) + Vnext(i));
            Vnext(t) = V(t) + H;
        }
    }
    ifSIGNAL(P(signal), triangle)
    {
        ifMETHOD(P(method), ExplicitEuler)
        {
            Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * std::floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)) * std::pow((numb)(-1), std::floor(((numb)4.0f * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)));
            Vnext(t) = V(t) + H;
            Vnext(x) = V(x) + H * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + Vnext(i));
            Vnext(y) = V(y) + H * (P(c) - P(d) * V(x) * V(x) - V(y));
            Vnext(z) = V(z) + H * (P(r) * (P(s) * (V(x) + P(e)) - V(z)));
        }

        ifMETHOD(P(method), ExplicitMidpoint)
        {
            numb imp = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * std::floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)) * std::pow((numb)(-1), std::floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)));
            numb tmp = V(t) + H * (numb)0.5;
            numb xmp = V(x) + H * (numb)0.5 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + imp);
            numb ymp = V(y) + H * (numb)0.5 * (P(c) - P(d) * V(x) * V(x) - V(y));
            numb zmp = V(z) + H * (numb)0.5 * (P(r) * (P(s) * (V(x) + P(e)) - V(z)));

            Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) - (numb)2.0 * std::floor(((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0)) * std::pow((numb)(-1), std::floor(((numb)4.0 * P(Ifreq) * (tmp - P(Idel)) + (numb)1.0) / (numb)2.0)));
            Vnext(t) = V(t) + H;
            Vnext(x) = V(x) + H * (ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + Vnext(i));
            Vnext(y) = V(y) + H * (P(c) - P(d) * xmp * xmp - ymp);
            Vnext(z) = V(z) + H * (P(r) * (P(s) * (xmp + P(e)) - zmp));
        }

        ifMETHOD(P(method), ExplicitRungeKutta4)
        {
            numb i1 = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * std::floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)) * std::pow((numb)(-1), std::floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)));
            numb kt1 = V(t) + (numb)0.5 * H;

            numb kx1 = V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + i1;
            numb ky1 = P(c) - P(d) * V(x) * V(x) - V(y);
            numb kz1 = P(r) * (P(s) * (V(x) + P(e)) - V(z));

            numb xmp = V(x) + (numb)0.5 * H * kx1;
            numb ymp = V(y) + (numb)0.5 * H * ky1;
            numb zmp = V(z) + (numb)0.5 * H * kz1;


            numb i2 = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (kt1 - P(Idel)) - (numb)2.0 * std::floor(((numb)4.0 * P(Ifreq) * (kt1 - P(Idel)) + (numb)1.0) / (numb)2.0)) * std::pow((numb)(-1), std::floor(((numb)4.0 * P(Ifreq) * (kt1 - P(Idel)) + (numb)1.0) / (numb)2.0)));
            numb kx2 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + i2;
            numb ky2 = P(c) - P(d) * xmp * xmp - ymp;
            numb kz2 = P(r) * (P(s) * (xmp + P(e)) - zmp);

            xmp = V(x) + (numb)0.5 * H * kx2;
            ymp = V(y) + (numb)0.5 * H * ky2;
            zmp = V(z) + (numb)0.5 * H * kz2;

            numb kx3 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + i2;
            numb ky3 = P(c) - P(d) * xmp * xmp - ymp;
            numb kz3 = P(r) * (P(s) * (xmp + P(e)) - zmp);
            Vnext(t) = V(t) + H;

            xmp = V(x) + H * kx3;
            ymp = V(y) + H * ky3;
            zmp = V(z) + H * kz3;

            numb i3 = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (Vnext(t) - P(Idel)) - (numb)2.0 * std::floor(((numb)4.0 * P(Ifreq) * (Vnext(t) - P(Idel)) + (numb)1.0) / (numb)2.0)) * std::pow((numb)(-1), std::floor(((numb)4.0 * P(Ifreq) * (Vnext(t) - P(Idel)) + (numb)1.0) / (numb)2.0)));
            numb kx4 = ymp - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - zmp + i3;
            numb ky4 = P(c) - P(d) * xmp * xmp - ymp;
            numb kz4 = P(r) * (P(s) * (xmp + P(e)) - zmp);

            Vnext(i) = i3;
            Vnext(x) = V(x) + H * (kx1 + (numb)2.0 * kx2 + (numb)2.0 * kx3 + kx4) / (numb)6.0;
            Vnext(y) = V(y) + H * (ky1 + (numb)2.0 * ky2 + (numb)2.0 * ky3 + ky4) / (numb)6.0;
            Vnext(z) = V(z) + H * (kz1 + (numb)2.0 * kz2 + (numb)2.0 * kz3 + kz4) / (numb)6.0;
        }

        ifMETHOD(P(method), VariableSymmetryCD)
        {
            numb h1 = (numb)0.5 * H - P(symmetry);
            numb h2 = (numb)0.5 * H + P(symmetry);


            Vnext(i) = P(Idc) + P(Iamp) * (((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) - (numb)2.0 * std::floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)) * std::pow((numb)(-1), std::floor(((numb)4.0 * P(Ifreq) * (V(t) - P(Idel)) + (numb)1.0) / (numb)2.0)));
            numb xmp = V(x) + h1 * (V(y) - P(a) * V(x) * V(x) * V(x) + P(b) * V(x) * V(x) - V(z) + Vnext(i));
            numb ymp = V(y) + h1 * (P(c) - P(d) * xmp * xmp - V(y));
            numb zmp = V(z) + h1 * (P(r) * (P(s) * (xmp + P(e)) - V(z)));

            Vnext(z) = (zmp + P(r) * P(s) * (xmp + P(e)) * h2) / (1 + P(r) * h2);
            Vnext(y) = (ymp + (P(c) - P(d) * xmp * xmp) * h2) / (1 + h2);

            Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * xmp * xmp * xmp + P(b) * xmp * xmp - Vnext(z) + Vnext(i));
            Vnext(x) = xmp + h2 * (Vnext(y) - P(a) * Vnext(x) * Vnext(x) * Vnext(x) + P(b) * Vnext(x) * Vnext(x) - Vnext(z) + Vnext(i));
            Vnext(t) = V(t) + H;
        }
    }
}

#undef name