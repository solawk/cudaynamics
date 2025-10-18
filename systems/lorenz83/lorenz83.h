#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_lorenz83 = 64;

__global__ void kernelProgram_lorenz83(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_lorenz83(numb* currentV, numb* nextV, numb* parameters, Computation* data);