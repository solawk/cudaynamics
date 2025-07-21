#include "max.h"

__device__ void MAX(Computation* data, MAX_Settings settings, int variation, void(* finiteDifferenceScheme)(numb*, numb*, numb*, numb), int offset)
{
    int variationStart = variation * CUDA_marshal.variationSize;
    int stepStart = variationStart;

    int var = settings.maxVariableIndex;
    numb maxValue = 0.0;

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        if (s == 0 || maxValue < CUDA_marshal.trajectory[stepStart + var]) maxValue = CUDA_marshal.trajectory[stepStart + var];
    }

    numb mapValue = maxValue;

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