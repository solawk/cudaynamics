#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_sprott14 = 64;

__global__ void kernelProgram_sprott14(Computation* data);

__device__ void finiteDifferenceScheme_sprott14(numb* currentV, numb* nextV, numb* parameters, numb h);