#include "max.h"

__device__ void MAX(Computation* data, uint64_t variation, void(* finiteDifferenceScheme)(numb*, numb*, numb*))
{
    uint64_t stepStart, variationStart = variation * CUDA_marshal.variationSize;
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(true);
    if (data->isHires) TRANSIENT_SKIP_NEW(finiteDifferenceScheme);

    MINMAX_Settings settings = CUDA_kernel.analyses.MINMAX;
    int minvar = settings.minVariableIndex;
    int maxvar = settings.maxVariableIndex;
    numb minValue = 0.0, maxValue = 0.0;
    numb prevMin, prevMax;

    for (int s = 0; s < CUDA_kernel.steps; s++)
    {
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;

        NORMAL_STEP_IN_ANALYSIS_IF_HIRES;

        if (CUDA_kernel.analyses.MINMAX.maximum.used)
        {
            prevMax = !data->isHires ? CUDA_marshal.trajectory[stepStart + maxvar] : variables[maxvar];
            if (s == 0 || maxValue < prevMax) maxValue = prevMax;
        }

        if (CUDA_kernel.analyses.MINMAX.minimum.used)
        {
            prevMin = !data->isHires ? CUDA_marshal.trajectory[stepStart + minvar] : variables[minvar];
            if (s == 0 || minValue > prevMin) minValue = prevMin;
        }
    }

    if (CUDA_kernel.mapWeight == 1.0f)
    {
        if (CUDA_kernel.analyses.MINMAX.maximum.used) CUDA_marshal.maps[indexPosition(settings.maximum.offset, 0)] = maxValue;
        if (CUDA_kernel.analyses.MINMAX.minimum.used) CUDA_marshal.maps[indexPosition(settings.minimum.offset, 0)] = minValue;
    }
    else
    {
        if (CUDA_kernel.analyses.MINMAX.maximum.used)
        {
            prevMax = CUDA_marshal.maps[indexPosition(settings.maximum.offset, 0)];
            CUDA_marshal.maps[indexPosition(settings.maximum.offset, 0)] = maxValue > prevMax ? maxValue : prevMax;
        }

        if (CUDA_kernel.analyses.MINMAX.minimum.used)
        {
            prevMin = CUDA_marshal.maps[indexPosition(settings.minimum.offset, 0)];
            CUDA_marshal.maps[indexPosition(settings.minimum.offset, 0)] = minValue < prevMin ? minValue : prevMin;
        }
    }
}