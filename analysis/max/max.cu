#include "max.h"

__device__ void MAX(Computation* data, int variation, void(* finiteDifferenceScheme)(numb*, numb*, numb*))
{
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(true);
    if (data->isHires) TRANSIENT_SKIP_NEW(finiteDifferenceScheme);

    MINMAX_Settings settings = CUDA_kernel.analyses.MINMAX;
    int var = settings.maxVariableIndex;
    numb minValue = 0.0, maxValue = 0.0;

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        NORMAL_STEP_IN_ANALYSIS_IF_HIRES;

        numb value = !data->isHires ? CUDA_marshal.trajectory[stepStart + var] : variables[var];
        if (s == 0 || maxValue < value) maxValue = value;
        if (s == 0 || minValue > value) minValue = value;
    }

    if (CUDA_kernel.mapWeight == 1.0f)
    {
        CUDA_marshal.maps[indexPosition(settings.maximum.offset, 0)] = maxValue;
        CUDA_marshal.maps[indexPosition(settings.minimum.offset, 0)] = minValue;
    }
    else
    {
        numb prevMax = CUDA_marshal.maps[indexPosition(settings.maximum.offset, 0)];
        numb prevMin = CUDA_marshal.maps[indexPosition(settings.minimum.offset, 0)];
        CUDA_marshal.maps[indexPosition(settings.maximum.offset, 0)] = maxValue > prevMax ? maxValue : prevMax;
        CUDA_marshal.maps[indexPosition(settings.minimum.offset, 0)] = minValue > prevMin ? minValue : prevMin;
    }
}