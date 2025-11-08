#pragma once
#include "cuda_runtime.h"
#include "../port.h"
#include "../numb.h"
#include "abstractSettings_struct.h"

struct MAX_Settings : AbstractAnalysisSettingsStruct
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