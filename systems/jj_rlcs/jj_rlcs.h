#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_jj_rlcs = 64;

__global__ void kernelProgram_jj_rlcs(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_jj_rlcs(numb* currentV, numb* nextV, numb* parameters);