#pragma once
#include "../analysis.h"
#include "../computation_struct.h"
#include "../mapData_struct.h"

#define MAX_SETTINGS_COUNT		1

struct MAX_Settings
{
	int maxVariableIndex;

	__device__ MAX_Settings(int _var)
	{
		maxVariableIndex = _var;
	}
};

__device__ void MAX(Computation* data, MAX_Settings settings, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*, numb), int offset);