#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_rlcs_jj = 64;

__global__ void kernelProgram_rlcs_jj(Computation* data);

__device__ void finiteDifferenceScheme_rlcs_jj(numb* currentV, numb* nextV, numb* parameters, numb h);