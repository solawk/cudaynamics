#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_mrlcs_jj = 64;

__global__ void kernelProgram_mrlcs_jj(Computation* data);

__device__ void finiteDifferenceScheme_mrlcs_jj(numb* currentV, numb* nextV, numb* parameters, numb h);