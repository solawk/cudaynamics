#include "lle.h"

__device__ void LLE(Computation* data, LLE_Settings settings, int variation, void(* finiteDifferenceScheme)(numb*, numb*, numb*, numb), int offset)
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

        finiteDifferenceScheme(LLE_array_temp, LLE_array, &(CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT]), CUDA_kernel.stepSize);

        if (s % L == 0)
        {
            numb norm = 0.0;
            for (int i = 0; i < MAX_LLE_NORM_VARIABLES; i++)
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
        numb existingValue = CUDA_marshal.maps[mapPosition] * data->bufferNo;
        CUDA_marshal.maps[mapPosition] = (existingValue + mapValue) / (data->bufferNo + 1);
    }
    else if (CUDA_kernel.mapWeight == 1.0f)
    {
        CUDA_marshal.maps[mapPosition] = mapValue;
    }
    else
    {
        CUDA_marshal.maps[mapPosition] = CUDA_marshal.maps[mapPosition] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
    }
}