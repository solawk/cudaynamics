#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_dadras = 64;

__global__ void kernelProgram_dadras(Computation* data);

__device__ void finiteDifferenceScheme_dadras(numb* currentV, numb* nextV, numb* parameters, numb h);