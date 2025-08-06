#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_lorenz = 64;

__global__ void kernelProgram_lorenz(Computation* data);

__device__ void finiteDifferenceScheme_lorenz(numb* currentV, numb* nextV, numb* parameters, numb h);