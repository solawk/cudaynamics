#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_sprottJm = 64;

__global__ void kernelProgram_sprottJm(Computation* data);

__device__ void finiteDifferenceScheme_sprottJm(numb* currentV, numb* nextV, numb* parameters);