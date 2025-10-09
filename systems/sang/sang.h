#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_sang = 64;

__global__ void kernelProgram_sang(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_sang(numb* currentV, numb* nextV, numb* parameters);