#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_rossler = 64;

__global__ void kernelProgram_rossler(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_rossler(numb* currentV, numb* nextV, numb* parameters);