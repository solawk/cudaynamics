#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_msprottj = 64;

__global__ void kernelProgram_msprottj(Computation* data);

__device__ void finiteDifferenceScheme_msprottj(numb* currentV, numb* nextV, numb* parameters, numb h);