#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_langford = 64;

__global__ void kernelProgram_langford(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_langford(numb* currentV, numb* nextV, numb* parameters, Computation* data);