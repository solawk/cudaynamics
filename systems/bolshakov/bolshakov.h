#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_bolshakov = 64;

__global__ void kernelProgram_bolshakov(Computation* data);

__device__ __forceinline__  void finiteDifferenceScheme_bolshakov(numb* currentV, numb* nextV, numb* parameters, Computation* data);