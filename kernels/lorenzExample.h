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

	enum MAPS { test, MAP_COUNT };
	extern const char* MAP_NAMES[];
	extern int MAP_X[];
	extern int MAP_Y[];

	extern int steps;
	extern bool executeOnLaunch;
	extern float stepSize;
	extern bool onlyShowLast;
}

const int THREADS_PER_BLOCK = 64;

//void preprocessing(float*);

__global__ void kernelProgram(float*, float*, float*, PreRanging*, int, float, int, float*);

//void postprocessing(float*);