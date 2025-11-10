#pragma once
#include "cuda_runtime.h"

#include "../port.h"
#include "../numb.h"
#include "abstractSettings_struct.h"

struct MINMAX_Settings : AbstractAnalysisSettingsStruct
{
	int maxVariableIndex;
	int minVariableIndex;

	Port maximum;
	Port minimum;

	MINMAX_Settings() 
	{
		maxVariableIndex = minVariableIndex = 0;
		maximum = minimum = Port();
	}

	void DisplaySettings()
	{
		DisplayVarSetting("Minimum variable", minVariableIndex);
		DisplayVarSetting("Maximum variable", maxVariableIndex);
	}
};