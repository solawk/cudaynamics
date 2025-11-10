#pragma once
#include "cuda_runtime.h"

#include "../port.h"
#include "../numb.h"
#include "abstractSettings_struct.h"

struct LLE_Settings : AbstractAnalysisSettingsStruct
{
	numb r;			// Initial deflection
	int L;			// Deflection observation duration (steps)
	int variableToDeflect;	// Variable to initially deflect
	int normVariables[4];	// Variables to count the norm with (-1 if not counting)

	Port LLE;

	LLE_Settings() 
	{
		r = (numb)0.01;
		L = 50;
		variableToDeflect = 0;
		normVariables[0] = 0;
		normVariables[1] = 1;
		normVariables[2] = 2;
		normVariables[3] = -1;

		LLE = Port();
	}

	void DisplaySettings()
	{
		DisplayNumbSetting("Initial deflection", r);
		DisplayIntSetting("Observation steps", L);
		DisplayVarSetting("Deflected variable", variableToDeflect);
		for (int i = 0; i < 4; i++) DisplayVarSetting("Norm variable " + std::to_string(i + 1), normVariables[i]);
	}
};