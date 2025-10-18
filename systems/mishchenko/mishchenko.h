#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_mishchenko = 64;

__global__ void kernelProgram_mishchenko(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_mishchenko(numb* currentV, numb* nextV, numb* parameters, Computation* data);