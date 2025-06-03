#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_chen = 64;

__global__ void kernelProgram_chen(Computation* data);

__device__ void finiteDifferenceScheme_chen(numb* currentV, numb* nextV, numb* parameters, numb h);