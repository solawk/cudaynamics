#include "main.h"
#include "lobby.cuh"

namespace attributes
{
    enum maps { LLE, MAX, MeanInterval, MeanPeak, Period };
}

__global__ void kernelLobby(Computation* data)
{
    int variation = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    int stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(false);

    data->kernelProgram(data);

    // Analysis

    if (M(LLE).toCompute)
    {
        LLE_Settings lle_settings(MS(LLE, 0), MS(LLE, 1), MS(LLE, 2));
        lle_settings.Use3DNorm();
        LLE(data, lle_settings, variation, data->kernelFDS, MO(LLE));
    }

    if (M(MAX).toCompute)
    {
        MAX_Settings max_settings(MS(MAX, 0));
        MAX(data, max_settings, variation, data->kernelFDS, MO(MAX));
    }

    if (M(Period).toCompute || M(MeanInterval).toCompute || M(MeanPeak).toCompute)
    {
        DBSCAN_Settings dbscan_settings(MS(Period, 0), MS(MeanInterval, 0), MS(Period, 1), MS(Period, 2), MS(MeanInterval, 1), MS(MeanInterval, 2), MS(MeanInterval, 3), MS(MeanInterval, 4),
            parameters[data->marshal.kernel.PARAM_COUNT]);
        Period(data, dbscan_settings, variation, data->kernelFDS, MO(Period), MO(MeanPeak), MO(MeanInterval));
    }
}