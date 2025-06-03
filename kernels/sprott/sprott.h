#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_sprott = 64;

__global__ void kernelProgram_sprott(Computation* data);

__device__ void finiteDifferenceScheme_sprott(numb* currentV, numb* nextV, numb* parameters, numb h);