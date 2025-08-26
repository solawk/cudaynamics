#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_fitznagumo = 64;

__global__ void kernelProgram_fitznagumo(Computation* data);

__device__ void finiteDifferenceScheme_fitznagumo(numb* currentV, numb* nextV, numb* parameters, numb h);