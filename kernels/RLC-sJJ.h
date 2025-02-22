#pragma once
#include "cuda_runtime.h"
#include "../objects.h"

namespace kernel
{
	extern const char* name;

	enum VARIABLES { sin_x0, x0, x1, x2, VAR_COUNT };
	extern const char* VAR_NAMES[];
	extern numb VAR_VALUES[];
	extern bool VAR_RANGING[];
	extern numb VAR_STEPS[];
	extern numb VAR_MAX[];
	extern int VAR_STEP_COUNTS[];

	enum PARAMETERS { p0, p1, p2, p3, p4, p5, PARAM_COUNT };
	extern const char* PARAM_NAMES[];
	extern numb PARAM_VALUES[];
	extern bool PARAM_RANGING[];
	extern numb PARAM_STEPS[];
	extern numb PARAM_MAX[];
	extern int PARAM_STEP_COUNTS[];

	enum MAPS { LLE, MAP_COUNT };
	extern const char* MAP_NAMES[];
	extern MapData MAP_DATA[];

	enum ANALYSIS { LLE_ANALYSIS, ANALYSIS_COUNT };
	extern const char* ANALYSIS_NAMES[];
	extern bool ANALYSIS_ENABLED[];

	extern int steps;
	extern bool executeOnLaunch;
	extern numb stepSize;
	extern bool onlyShowLast;
}

const int THREADS_PER_BLOCK = 64;

__global__ void kernelProgram(numb*, numb*, numb*, MapData*, PreRanging*, int, numb, int, numb*);

__device__ void finiteDifferenceScheme(numb* currentV, numb* nextV, numb* parameters, numb h);