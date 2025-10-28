#include "lle.h"
#pragma warning(push)
#pragma warning(disable:6385)

__device__ void LLE(Computation* data, LLE_Settings settings, int variation, void(* finiteDifferenceScheme)(numb*, numb*, numb*), int offset)
{
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(true);
    if (data->isHires) TRANSIENT_SKIP_NEW(finiteDifferenceScheme);

    numb LLE_array[MAX_ATTRIBUTES]{ 0 }; // The deflected trajectory
    numb LLE_array_next[MAX_ATTRIBUTES]; // Buffer for the next step of the deflected trajectory
    numb LLE_value = 0.0f;

    for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
        if (!data->isHires)     LLE_array[i] = CUDA_marshal.trajectory[variationStart + i];
        else                    LLE_array[i] = variables[i];

    numb r = settings.r;
    int L = settings.L;
    LLE_array[settings.variableToDeflect] += r;
    int stepParamIndex = CUDA_kernel.PARAM_COUNT - 1;

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        // Normal step, if hi-res
        NORMAL_STEP_IN_ANALYSIS_IF_HIRES;

        // Deflected step
        finiteDifferenceScheme(LLE_array, LLE_array_next, &(parameters[0]));

        for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
            LLE_array[i] = LLE_array_next[i];

        // LLE calculations
        if ((s + 1) % L == 0)
        {
            numb norm = 0.0;
            for (int i = 0; i < MAX_LLE_NORM_VARIABLES; i++)
            {
                if (settings.normVariables[i] == -1) break;

                numb x1 = LLE_array[settings.normVariables[i]];
                numb x2 = !data->isHires ? CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT + settings.normVariables[i]] : variables[settings.normVariables[i]];
                norm += (x2 - x1) * (x2 - x1);
            }

            norm = sqrt(norm);

            numb growth = norm / r; // How many times the deflection has grown
            if (growth > 0.0f)
                LLE_value += log(growth) / H_BRANCH(
                    parameters[stepParamIndex],
                    !data->isHires ? CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT - 1] : variables[CUDA_kernel.VAR_COUNT - 1]
                );

            // Reset
            for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
                if (!data->isHires) LLE_array[i] = CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT + i] + (LLE_array[i] - CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT + i]) / growth;
                else LLE_array[i] = variables[i] + (LLE_array[i] - variables[i]) / growth;
        }
    }

    numb mapValue = LLE_value / (CUDA_kernel.steps + 1);

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

#pragma warning(pop)