#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_mixed = 64;

__global__ void kernelProgram_mixed(Computation* data);

__device__ __forceinline__  void finiteDifferenceScheme_mixed(numb* currentV, numb* nextV, numb* parameters);