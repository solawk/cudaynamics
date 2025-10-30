#pragma once
#include "cuda_runtime.h"
#include "cuda_macros.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <chrono>
#include <wtypes.h>

const int THREADS_PER_BLOCK_halvorsen = 64;

__global__ void kernelProgram_halvorsen(Computation* data);

__device__ __forceinline__ void finiteDifferenceScheme_halvorsen(numb* currentV, numb* nextV, numb* parameters);