#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_rabinovich_fabrikant = 64;

__global__ void kernelProgram_rabinovich_fabrikant(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_rabinovich_fabrikant(numb* currentV, numb* nextV, numb* parameters);