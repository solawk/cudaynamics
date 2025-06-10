#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_lorenzMod = 64;

__global__ void kernelProgram_lorenzMod(Computation* data);

__device__ void finiteDifferenceScheme_lorenzMod(numb* currentV, numb* nextV, numb* parameters, numb h);