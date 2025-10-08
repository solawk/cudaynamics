#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_hindmarsh_rose = 64;

__global__ void kernelProgram_hindmarsh_rose(Computation* data);

__device__ __forceinline__  void finiteDifferenceScheme_hindmarsh_rose(numb* currentV, numb* nextV, numb* parameters);