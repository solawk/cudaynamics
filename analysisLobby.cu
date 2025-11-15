#include "analysisLobby.cuh"

__device__ void AnalysisLobby(Computation* data, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int variation)
{
    if (CUDA_kernel.analyses.LLE.toCompute)
    {
        LLE(data, variation, finiteDifferenceScheme);
    }

    if (CUDA_kernel.analyses.MINMAX.toCompute)
    {
        MAX(data, variation, finiteDifferenceScheme);
    }

    if (CUDA_kernel.analyses.PERIOD.toCompute)
    {
        Period(data, variation, finiteDifferenceScheme);
    }
}