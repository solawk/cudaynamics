#pragma once
#include <kernels_common.h>

const int THREADS_PER_BLOCK_vnm = 64;

__global__ void kernelProgram_vnm(Computation* data);

__device__ void finiteDifferenceScheme_vnm(numb* currentV, numb* nextV, numb* parameters, numb h);