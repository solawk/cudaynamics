#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_izhikevich = 64;

__global__ void kernelProgram_izhikevich(Computation* data);

__device__ __forceinline__  void finiteDifferenceScheme_izhikevich(numb* currentV, numb* nextV, numb* parameters);