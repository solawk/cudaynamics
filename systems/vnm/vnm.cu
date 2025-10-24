#include "main.h"
#include "vnm.h"

namespace attributes
{
    enum variables { x, y, z, t, w };
    enum parameters { timePerSystem, symmetry, method, COUNT };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelProgram_vnm(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;            // Variation (parameter combination) index
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    numb variables[MAX_ATTRIBUTES];
    numb variablesNext[MAX_ATTRIBUTES];
    numb parameters[MAX_ATTRIBUTES];
    LOAD_ATTRIBUTES(false);

    // Custom area (usually) starts here

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_vnm);

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_vnm(FDS_ARGUMENTS);
        RECORD_STEP;
    }

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, &finiteDifferenceScheme_vnm, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, &finiteDifferenceScheme_vnm, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBSCAN_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4),
            H);
        Period(data, dbscan_settings, variation, &finiteDifferenceScheme_vnm, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}

__device__ __forceinline__ void finiteDifferenceScheme_vnm(numb* currentV, numb* nextV, numb* parameters)
{
    int system = (int)(V(t) / P(timePerSystem)) % 4;

    numb xmp, ymp, zmp;

    // Hard bounds
    if (V(x) > 300.0) V(x) = 300.0;
    if (V(y) > 300.0) V(y) = 300.0;
    if (V(z) > 300.0) V(z) = 300.0;
    if (V(x) < -300.0) V(x) = -300.0;
    if (V(y) < -300.0) V(y) = -300.0;
    if (V(z) < -300.0) V(z) = -300.0;

    switch (system)
    {
    case 0:
        // Lorenz (working)
        xmp = V(x) + H * 0.5 * (10.0 * (V(y) - V(x)));
        ymp = V(y) + H * 0.5 * (V(x) * (28.0 - V(z)) - V(y));
        zmp = V(z) + H * 0.5 * (V(x) * V(y) - 2.66667 * V(z));
        Vnext(x) = V(x) + H * (10.0 * (ymp - xmp));
        Vnext(y) = V(y) + H * (xmp * (28.0 - zmp) - ymp);
        Vnext(z) = V(z) + H * (xmp * ymp - 2.66667 * zmp);
        Vnext(w) = Vnext(x);
        break;
    case 1:
        // Chen (working)
        xmp = V(x) + 0.5 * H * 3.0 * (5.0 * V(x) - V(y) * V(z) / 2.0);
        ymp = V(y) + 0.5 * H * 3.0 * (-10.0 * V(y) + V(x) * V(z) / 2.0);
        zmp = V(z) + 0.5 * H * 3.0 * (-0.38 * V(z) / 2.0 + V(x) * V(y) / 3.0);
        Vnext(x) = V(x) + H * 3.0 * (5.0 * xmp - ymp * zmp / 2.0);
        Vnext(y) = V(y) + H * 3.0 * (-10.0 * ymp + xmp * zmp / 2.0);
        Vnext(z) = V(z) + H * 3.0 * (-0.38 * zmp / 2.0 + xmp * ymp / 3.0);
        Vnext(w) = Vnext(x);
        break;
    case 2:
        // Thomas (working)
        xmp = V(x) + H * 50.0 * 0.5 * (sin(V(y) / 5.0) - 0.208186 * (V(x) / 5.0));
        ymp = V(y) + H * 50.0 * 0.5 * (sin(V(z) / 5.0 - 6.0) - 0.208186 * (V(y) / 5.0));
        zmp = V(z) + H * 50.0 * 0.5 * (sin(V(x) / 5.0) - 0.208186 * (V(z) / 5.0 - 6.0));
        Vnext(x) = V(x) + H * 50.0 * (sin(ymp / 5.0) - 0.208186 * (xmp / 5.0));
        Vnext(y) = V(y) + H * 50.0 * (sin(zmp / 5.0 - 6.0) - 0.208186 * (ymp / 5.0));
        Vnext(z) = V(z) + H * 50.0 * (sin(xmp / 5.0) - 0.208186 * (zmp / 5.0 - 6.0));
        Vnext(w) = Vnext(x);
        break;
    case 3:
        // Coullet (working, requires hard boundary)
        xmp = V(x) + 0.5 * H * 250.0 * V(y) / 20.0;
        ymp = V(y) + 0.5 * H * 250.0 * ((V(z) - 25.0) / 20.0);
        zmp = V(z) + 0.5 * H * 250.0 * (0.8 * (V(x) / 20.0) - 1.1 * (V(y) / 20.0) - 0.45 * ((V(z) - 25.0) / 20.0) - (V(x) / 20.0) * (V(x) / 20.0) * (V(x) / 20.0));
        Vnext(x) = V(x) + H * 250.0 * ymp / 20.0;
        Vnext(y) = V(y) + H * 250.0 * ((zmp - 25.0) / 20.0);
        Vnext(z) = V(z) + H * 250.0 * (0.8 * (xmp / 20.0) - 1.1 * (ymp / 20.0) - 0.45 * ((zmp - 25.0) / 20.0) - (xmp / 20.0) * (xmp / 20.0) * (xmp / 20.0));
        Vnext(w) = Vnext(x);
        break;

        // RLC-sJJ (slow and small)
        // x1 -> x
        // x2 -> y
        // x3 -> z
        // sin_x1 -> w 
        /*numb x1mp = fmodf(V(x) + H * 5.0 * 0.5 * V(y), 2.0f * 3.141592f);
        numb x2mp = V(y) + H * 5.0 * 0.5 * (1.0f / 0.707) * (1.25 - ((V(y) > 6.9) ? 0.367 : 0.0478) * V(y) - sinf(V(x)) - V(z) / 5.0);
        numb x3mp = V(z) + H * 5.0 * 0.5 * (1.0f / 29.215) * (V(y) - V(z) / 5.0);
        Vnext(x) = fmodf(V(x) + H * 5.0 * x2mp, 2.0f * 3.141592f);
        Vnext(y) = V(y) + H * 5.0 * (1.0f / 0.707) * (1.25 - ((x2mp > 6.9) ? 0.367 : 0.0478) * x2mp - sinf(x1mp) - x3mp / 5.0);
        Vnext(z) = V(z) + H * 5.0 * (1.0f / 29.215) * (x2mp - x3mp / 5.0);
        Vnext(w) = sinf(Vnext(x));
        break;*/
    }

    Vnext(t) = V(t) + H;
}