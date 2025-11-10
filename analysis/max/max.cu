#include "max.h"

__device__ void MAX(Computation* data, int variation, void(* finiteDifferenceScheme)(numb*, numb*, numb*), int offset)
{
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(true);
    if (data->isHires) TRANSIENT_SKIP_NEW(finiteDifferenceScheme);

    int var = CUDA_kernel.analyses.MINMAX.maxVariableIndex;
    numb maxValue = 0.0;

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        NORMAL_STEP_IN_ANALYSIS_IF_HIRES;

        numb value = !data->isHires ? CUDA_marshal.trajectory[stepStart + var] : variables[var];
        if (s == 0 || maxValue < value) maxValue = value;
    }

    numb mapValue = maxValue;

    if (CUDA_kernel.mapWeight == 1.0f)
    {
        CUDA_marshal.maps[mapPosition] = mapValue;
    }
    else
    {
        CUDA_marshal.maps[mapPosition] = mapValue > CUDA_marshal.maps[mapPosition] ? mapValue : CUDA_marshal.maps[mapPosition];
    }
}