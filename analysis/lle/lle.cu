#include "lle.h"
#pragma warning(push)
#pragma warning(disable:6385)

__host__ __device__ void LLE(Computation* data, uint64_t variation, void(* finiteDifferenceScheme)(numb*, numb*, numb*))
{
    uint64_t stepStart, variationStart = variation * CUDA_marshal.variationSize;
    int stepParamIndex = CUDA_kernel.PARAM_COUNT - 1;
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(true);
    if (data->isHires) TRANSIENT_SKIP_NEW(finiteDifferenceScheme);

    numb LLE_array[MAX_ATTRIBUTES]{ 0 }; // The deflected trajectory
    numb LLE_array_next[MAX_ATTRIBUTES]; // Buffer for the next step of the deflected trajectory
    numb LLE_value = 0.0f;

    for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
        if (!data->isHires)     LLE_array[i] = CUDA_marshal.trajectory[variationStart + i];
        else                    LLE_array[i] = variables[i];

    LLE_Settings settings = CUDA_kernel.analyses.LLE;
    numb r = settings.r;
    int L = settings.L;
    LLE_array[settings.variableToDeflect] += r;

    numb sfloat = (numb)0.0;
    int scounter = 0;
    int lastAnalysedStep = -1;
    int analysedSteps = 0;

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        // Normal step, if hi-res
        NORMAL_STEP_IN_ANALYSIS_IF_HIRES;

        // Deflected step
        finiteDifferenceScheme(LLE_array, LLE_array_next, &(parameters[0]));
        for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
            LLE_array[i] = LLE_array_next[i];

        sfloat += (numb)CUDA_kernel.targetSteps / CUDA_kernel.steps;
        if (sfloat >= (numb)1.0)
        {
            sfloat -= floor(sfloat);
            scounter++;

            // LLE calculations
            if (scounter % L == 0 && scounter > 0)
            {
                int64_t comparedIndex = variationStart + CUDA_kernel.VAR_COUNT * scounter;

                numb norm = 0.0;
                for (int i = 0; i < 4; i++)
                {
                    if (settings.normVariables[i] == -1) break;

                    numb x1 = LLE_array[settings.normVariables[i]];
                    numb x2 = !data->isHires ? CUDA_marshal.trajectory[comparedIndex + settings.normVariables[i]] : variables[settings.normVariables[i]];
                    norm += (x2 - x1) * (x2 - x1);
                }

                norm = sqrt(norm);
                numb growth = norm / r; // How many times the deflection has grown
                if (growth > 0.0f)
                    LLE_value += log(growth) / parameters[stepParamIndex];

                analysedSteps += s - lastAnalysedStep;
                lastAnalysedStep = s;

                // Reset
                for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++)
                {
                    if (!data->isHires)
                        LLE_array[i] = CUDA_marshal.trajectory[comparedIndex + i] + (LLE_array[i] - CUDA_marshal.trajectory[comparedIndex + i]) / growth;
                    else
                        LLE_array[i] = variables[i] + (LLE_array[i] - variables[i]) / growth;
                }
            }
        }
    }

    if (CUDA_kernel.analyses.LLE.LLE.used)
    {
        numb mapValue = LLE_value / analysedSteps;
        if (CUDA_kernel.mapWeight == 0.0f)
        {
            CUDA_marshal.maps[indexPosition(settings.LLE.offset, 0)] = (CUDA_marshal.maps[indexPosition(settings.LLE.offset, 0)] * data->bufferNo + mapValue) / (data->bufferNo + 1);
        }
        else if (CUDA_kernel.mapWeight == 1.0f)
        {
            CUDA_marshal.maps[indexPosition(settings.LLE.offset, 0)] = mapValue;
        }
        else
        {
            CUDA_marshal.maps[indexPosition(settings.LLE.offset, 0)] = CUDA_marshal.maps[indexPosition(settings.LLE.offset, 0)] * (1.0f - CUDA_kernel.mapWeight) + mapValue * CUDA_kernel.mapWeight;
        }
    }
}

#pragma warning(pop)