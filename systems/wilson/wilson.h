#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_wilson = 64;

__global__ void kernelProgram_wilson(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_wilson(numb* currentV, numb* nextV, numb* parameters);