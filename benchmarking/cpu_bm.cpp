#include "cpu_bm.h"

namespace attributes
{
    enum variables { x, y, z };
    enum parameters { sigma, rho, beta, stepsize, symmetry, method };
    enum methods { ExplicitEuler, ExplicitMidpoint, ExplicitRungeKutta4, VariableSymmetryCD };
    enum maps { LLE, MAX };
}

void finiteDifferenceScheme_lorenz_cpu(numb* currentV, numb* nextV, numb* parameters)
{
    ifMETHOD(P(method), ExplicitEuler)
    {
        Vnext(x) = V(x) + P(stepsize) * (P(sigma) * (V(y) - V(x)));
        Vnext(y) = V(y) + P(stepsize) * (V(x) * (P(rho) - V(z)) - V(y));
        Vnext(z) = V(z) + P(stepsize) * (V(x) * V(y) - P(beta) * V(z));
    }

    ifMETHOD(P(method), ExplicitMidpoint)
    {
        numb xmp = V(x) + P(stepsize) * 0.5f * (P(sigma) * (V(y) - V(x)));
        numb ymp = V(y) + P(stepsize) * 0.5f * (V(x) * (P(rho) - V(z)) - V(y));
        numb zmp = V(z) + P(stepsize) * 0.5f * (V(x) * V(y) - P(beta) * V(z));

        Vnext(x) = V(x) + P(stepsize) * (P(sigma) * (ymp - xmp));
        Vnext(y) = V(y) + P(stepsize) * (xmp * (P(rho) - zmp) - ymp);
        Vnext(z) = V(z) + P(stepsize) * (xmp * ymp - P(beta) * zmp);
    }

    ifMETHOD(P(method), ExplicitRungeKutta4)
    {
        numb kx1 = P(sigma) * (V(y) - V(x));
        numb ky1 = V(x) * (P(rho) - V(z)) - V(y);
        numb kz1 = V(x) * V(y) - P(beta) * V(z);

        numb xmp = V(x) + 0.5f * P(stepsize) * kx1;
        numb ymp = V(y) + 0.5f * P(stepsize) * ky1;
        numb zmp = V(z) + 0.5f * P(stepsize) * kz1;

        numb kx2 = P(sigma) * (ymp - xmp);
        numb ky2 = xmp * (P(rho) - zmp) - ymp;
        numb kz2 = xmp * ymp - P(beta) * zmp;

        xmp = V(x) + 0.5f * P(stepsize) * kx2;
        ymp = V(y) + 0.5f * P(stepsize) * ky2;
        zmp = V(z) + 0.5f * P(stepsize) * kz2;

        numb kx3 = P(sigma) * (ymp - xmp);
        numb ky3 = xmp * (P(rho) - zmp) - ymp;
        numb kz3 = xmp * ymp - P(beta) * zmp;

        xmp = V(x) + P(stepsize) * kx3;
        ymp = V(y) + P(stepsize) * ky3;
        zmp = V(z) + P(stepsize) * kz3;

        numb kx4 = P(sigma) * (ymp - xmp);
        numb ky4 = xmp * (P(rho) - zmp) - ymp;
        numb kz4 = xmp * ymp - P(beta) * zmp;

        Vnext(x) = V(x) + P(stepsize) * (kx1 + 2.0f * kx2 + 2.0f * kx3 + kx4) / 6.0f;
        Vnext(y) = V(y) + P(stepsize) * (ky1 + 2.0f * ky2 + 2.0f * ky3 + ky4) / 6.0f;
        Vnext(z) = V(z) + P(stepsize) * (kz1 + 2.0f * kz2 + 2.0f * kz3 + kz4) / 6.0f;
    }

    ifMETHOD(P(method), VariableSymmetryCD)
    {
        numb h1 = 0.5f * P(stepsize) - P(symmetry);
        numb h2 = 0.5f * P(stepsize) + P(symmetry);

        numb xmp = V(x) + h1 * (P(sigma) * (V(y) - V(x)));
        numb ymp = V(y) + h1 * (V(x) * (P(rho) - V(z)) - V(y));
        numb zmp = V(z) + h1 * (V(x) * V(y) - P(beta) * V(z));

        Vnext(z) = (zmp + xmp * ymp * h2) / (1.0f + P(beta) * h2);
        Vnext(y) = (ymp + xmp * (P(rho) - Vnext(z)) * h2) / (1.0f + h2);
        Vnext(x) = (xmp + P(sigma) * Vnext(y) * h2) / (1.0f + P(sigma) * h2);
    }
}

__device__ void LLE_cpu(Computation* data, LLE_Settings settings, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int offset)
{
    int variationStart = variation * CUDA_marshal.variationSize;
    int stepStart = variationStart;

    numb LLE_array[MAX_ATTRIBUTES]; // The deflected trajectory
    numb LLE_array_temp[MAX_ATTRIBUTES]; // Buffer for the next step of the deflected trajectory
    numb LLE_value = 0.0f;

    for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
        LLE_array[i] = CUDA_marshal.trajectory[stepStart + i];

    numb r = settings.r;
    int L = settings.L;
    LLE_array[settings.variableToDeflect] += r;

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
            LLE_array_temp[i] = LLE_array[i];

        finiteDifferenceScheme(LLE_array_temp, LLE_array, &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]));

        if (s % L == 0)
        {
            numb norm = 0.0;
            for (int i = 0; i < 4; i++)
            {
                if (settings.normVariables[i] == -1) break;

                numb x1 = LLE_array[settings.normVariables[i]];
                numb x2 = CUDA_marshal.trajectory[stepStart + settings.normVariables[i]];
                norm += (x2 - x1) * (x2 - x1);
            }

            norm = sqrt(norm);

            numb growth = norm / r; // How many times the deflection has grown
            if (growth > 0.0f) LLE_value += log(growth);

            // Reset
            for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
                LLE_array[i] = CUDA_marshal.trajectory[stepStart + i] + (LLE_array[i] - CUDA_marshal.trajectory[stepStart + i]) / growth;
        }
    }

    numb mapValue = LLE_value / ((CUDA_kernel.steps + 1) * CUDA_kernel.stepSize);

    if (CUDA_kernel.mapWeight == 0.0f)
    {
        CUDA_marshal.maps[mapPosition] = (CUDA_marshal.maps[mapPosition] * data->bufferNo + mapValue) / (data->bufferNo + 1);
    }
    else if (CUDA_kernel.mapWeight == 1.0f)
    {
        CUDA_marshal.maps[mapValueAt(0)] = mapValue;
    }
    else
    {
        CUDA_marshal.maps[mapPosition] = CUDA_marshal.maps[mapPosition] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
    }
}

void lorenz_cpu(Computation* data, int variation)
{
    int variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    int stepStart = variationStart;                         // Start index for the current modelling step
    if (variation >= CUDA_marshal.totalVariations) return;

    // Custom area (usually) starts here

    //TRANSIENT_SKIP_NEW(finiteDifferenceScheme_lorenz_cpu);

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        finiteDifferenceScheme_lorenz_cpu(&(CUDA_marshal.trajectory[stepStart]),
            &(CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT]),
            &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]));
    }

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), (int)MS(LLE, 1), (int)MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE_cpu(data, lle_settings, variation, &finiteDifferenceScheme_lorenz_cpu, (int)(MO(LLE) + (!data->isHires ? 0 : data->startVariationInCurrentExecute)));
    }
}

void cpu_execute(Computation* data, bool openmp)
{
	int variations = CUDA_marshal.totalVariations;

    if (!openmp)
    {
        for (int v = 0; v < variations; v++)
        {
            lorenz_cpu(data, v);
        }
    }
    else
    {
        if (!data->isHires)
        {
#pragma omp parallel for
            for (int v = 0; v < variations; v++)
            {
                lorenz_cpu(data, v);
            }
        }
        else
        {
#pragma omp parallel for
            for (int v = 0; v < data->variationsInCurrentExecute; v++)
            {
                lorenz_cpu(data, v);
            }
        }
    }
}