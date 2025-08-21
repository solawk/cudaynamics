#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_fitzhugh_nagumo = 64;

__global__ void kernelProgram_fitzhugh_nagumo(Computation* data);

__device__ void finiteDifferenceScheme_fitzhugh_nagumo(numb* currentV, numb* nextV, numb* parameters, numb h);