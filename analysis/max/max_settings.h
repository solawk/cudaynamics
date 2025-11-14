#pragma once
#include "cuda_runtime.h"
#include "../port.h"
#include "../numb.h"
#include "abstractSettings_struct.h"

struct MINMAX_Settings : AbstractAnalysisSettingsStruct
{
	int minVariableIndex;
	int maxVariableIndex;

	Port minimum;
	Port maximum;

	MINMAX_Settings() 
	{
		maxVariableIndex = minVariableIndex = 0;
		maximum = minimum = Port();
	}

	void DisplaySettings(std::vector<Attribute>& variables)
	{
		DisplayVarSetting("Minimum variable", minVariableIndex, variables);
		DisplayVarSetting("Maximum variable", maxVariableIndex, variables);
	}

	bool setup(std::vector<std::string> s)
	{
		if (!isMapSetupOfCorrectLength(s, 2)) return false;

		minVariableIndex = s2i(s[0]);
		maxVariableIndex = s2i(s[1]);

		return true;
	}
};