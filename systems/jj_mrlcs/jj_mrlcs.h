#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_jj_mrlcs = 64;

__global__ void kernelProgram_jj_mrlcs(Computation* data);

__device__ void finiteDifferenceScheme_jj_mrlcs(numb* currentV, numb* nextV, numb* parameters);