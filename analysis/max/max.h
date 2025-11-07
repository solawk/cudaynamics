#pragma once
#include "../analysis.h"
#include "../computation_struct.h"

struct MAX_Settings
{
	int maxVariableIndex;

	Port maximum;
	Port minimum;

	__device__ MAX_Settings(int _var)
	{
		maxVariableIndex = _var;
	}
};

__device__ void MAX(Computation* data, MAX_Settings settings, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int offset);