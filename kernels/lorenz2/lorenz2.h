#pragma once
#include "cuda_runtime.h"
#include "cuda_macros.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <chrono>
#include <wtypes.h>

const int THREADS_PER_BLOCK_lorenz2 = 64;

__global__ void kernelProgram_lorenz2(Computation* data);

__device__ void finiteDifferenceScheme_lorenz2(numb* currentV, numb* nextV, numb* parameters, numb h);