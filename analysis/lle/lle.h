#pragma once
#include "../analysis.h"
#include "../computation_struct.h"
#include "../mapData_struct.h"

#define MAX_LLE_NORM_VARIABLES	4
#define LLE_SETTINGS_COUNT		(3 + MAX_LLE_NORM_VARIABLES)

struct LLE_Settings
{
	numb r;			// Initial deflection
	int L;			// Deflection observation duration (steps)

	int variableToDeflect;	// Variable to initially deflect
	int normVariables[MAX_LLE_NORM_VARIABLES];	// Variables to count the norm with (-1 if not counting)

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

__device__ void LLE(Computation* data, LLE_Settings settings, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int offset);