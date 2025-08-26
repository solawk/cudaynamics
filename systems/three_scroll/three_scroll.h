#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_three_scroll = 64;

__global__ void kernelProgram_three_scroll(Computation* data);

__device__ void finiteDifferenceScheme_three_scroll(numb* currentV, numb* nextV, numb* parameters, numb h);