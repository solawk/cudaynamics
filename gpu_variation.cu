#include "gpu_variation.cuh"

__host__ __device__ uint64_t gpu_variation()
{
    //return (blockIdx.x * blockDim.x) + threadIdx.x;
    return (uint64_t)0;
}