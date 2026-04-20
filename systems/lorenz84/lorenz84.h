#pragma once
#include <kernels_common.h>

#define name lorenz84

__global__ void gpu_wrapper_(name)(Computation* data, uint64_t variation);

__host__ __device__ void kernelProgram_(name)(Computation* data, uint64_t variation);

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters, PerThread* pt);

#undef name