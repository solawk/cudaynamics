#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_thomas = 64;

__global__ void kernelProgram_thomas(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_thomas(numb* currentV, numb* nextV, numb* parameters);