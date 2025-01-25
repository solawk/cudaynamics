#pragma once
#include "cuda_runtime.h"
#include "../objects.h"

namespace kernel
{
	extern const char* name;

	enum VARIABLES { x, y, z, VAR_COUNT };
	extern const char* VAR_NAMES[];
	extern float VAR_VALUES[];
	extern bool VAR_RANGING[];
	extern float VAR_STEPS[];
	extern float VAR_MAX[];
	extern int VAR_STEP_COUNTS[];

	enum PARAMETERS { sigma, rho, beta, PARAM_COUNT };
	extern const char* PARAM_NAMES[];
	extern float PARAM_VALUES[];
	extern bool PARAM_RANGING[];
	extern float PARAM_STEPS[];
	extern float PARAM_MAX[];
	extern int PARAM_STEP_COUNTS[];

	enum MAPS { LLE, MAP_COUNT };
	extern const char* MAP_NAMES[];
	extern MapData MAP_DATA[];

	enum ANALYSIS { LLE_ANALYSIS, ANALYSIS_COUNT };
	extern const char* ANALYSIS_NAMES[];
	extern bool ANALYSIS_ENABLED[];

	extern int steps;
	extern bool executeOnLaunch;
	extern float stepSize;
	extern bool onlyShowLast;
}

const int THREADS_PER_BLOCK = 64;

__global__ void kernelProgram(float*, float*, float*, MapData*, PreRanging*, int, float, int, float*);

__device__ void finiteDifferenceScheme(float* currentV, float* nextV, float* parameters, float h);