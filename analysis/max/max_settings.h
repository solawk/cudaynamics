#pragma once
#include "cuda_runtime.h"
#include "../port.h"
#include "../numb.h"

struct MAX_Settings
{
	int maxVariableIndex;

	Port maximum;
	Port minimum;

	MAX_Settings()
	{

	}

	__device__ MAX_Settings(int _var)
	{
		maxVariableIndex = _var;
	}
};