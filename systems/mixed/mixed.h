#pragma once
#include <kernels_common.h>

#define name mixed

const int THREADS_PER_BLOCK_(name) = 64;

__global__ void kernelProgram_(name)(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters);

#undef name