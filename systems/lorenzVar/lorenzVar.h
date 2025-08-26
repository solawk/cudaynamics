#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_lorenzVar = 64;

__global__ void kernelProgram_lorenzVar(Computation* data);

__device__ void finiteDifferenceScheme_lorenzVar(numb* currentV, numb* nextV, numb* parameters);