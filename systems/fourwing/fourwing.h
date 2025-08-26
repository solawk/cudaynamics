#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_fourwing = 64;

__global__ void kernelProgram_fourwing(Computation* data);

__device__ void finiteDifferenceScheme_fourwing(numb* currentV, numb* nextV, numb* parameters);