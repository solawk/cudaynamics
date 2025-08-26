#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_nngu = 64;

__global__ void kernelProgram_nngu(Computation* data);

__device__ void finiteDifferenceScheme_nngu(numb* currentV, numb* nextV, numb* parameters);