#pragma once
#include "cuda_runtime.h"
#include "../port.h"
#include "../numb.h"
#include "abstractSettings_struct.h"

constexpr int MAX_LLE_NORM_VARIABLES = 4;

struct LLE_Settings : AbstractAnalysisSettingsStruct
{
	numb r;			// Initial deflection
	int L;			// Deflection observation duration (steps)

	int variableToDeflect;	// Variable to initially deflect
	int normVariables[MAX_LLE_NORM_VARIABLES];	// Variables to count the norm with (-1 if not counting)

	LLE_Settings() {}

	__device__ LLE_Settings(numb _r, int _L, int _varToDeflect)
	{
		r = _r;
		L = _L;
		variableToDeflect = _varToDeflect;
		normVariables[0] = -1;
	}

	__device__ void Use3DNorm()
	{
		normVariables[0] = 0;
		normVariables[1] = 1;
		normVariables[2] = 2;
		normVariables[3] = -1;
	}
};