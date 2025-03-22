#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "objects.h"
#include "computation_struct.h"
#include "cuda_macros.h"

// Variation to steps
__device__ void variation2Steps(int* variation, int* steps, Kernel* kernel);

// Steps to variation
void steps2Variation(int* variation, int* steps, Kernel* kernel);