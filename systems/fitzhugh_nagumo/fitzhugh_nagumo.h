#pragma once
#include <kernels_common.h>

#define name fitzhugh_nagumo

const int THREADS_PER_BLOCK_(name) = 64;

__global__ void kernelProgram_(name)(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters);

#undef name