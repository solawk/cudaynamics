#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <chrono>
#include <wtypes.h>
#include "analysisHeaders.h"
#include "analysisSettingsHeaders.h"
#include "analysisLobby.cuh"
#include "gpu_variation.cuh"
#include "cuda_macros.h"

#define CONCAT(a, b) a##b
#define THREADS_PER_BLOCK_(name) CONCAT(THREADS_PER_BLOCK_, name)
#define kernelProgram_(name) CONCAT(kernelProgram_, name)
#define finiteDifferenceScheme_(name) CONCAT(finiteDifferenceScheme_, name)
#define gpu_wrapper_(name)	CONCAT(gpu_wrapper_, name)